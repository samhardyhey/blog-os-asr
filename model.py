import logging
import os
import pickle
import string
import tempfile
from copy import deepcopy
from math import ceil, floor
from pathlib import Path

import nemo.collections.asr as nemo_asr
import pandas as pd
import torch
from ctcdecode import CTCBeamDecoder
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from pydub import AudioSegment
from pydub.silence import detect_silence

logging.getLogger("nemo_logger").setLevel(logging.ERROR)
asr_logger = logging.getLogger("asr")
asr_logger.setLevel(logging.INFO)


class OsASR:
    def __init__(self, model_path=None, use_gpu=True, **kwargs):
        # define default models to use
        self.default_dia_model = "pyannote/pyannote-audio"
        self.default_asr_model = "stt_en_conformer_ctc_small"  #'QuartzNet15x5Base-En'
        self.default_punct_model = "punctuation_en_bert"
        self.device = 0 if use_gpu else -1
        # load
        self.model_path = None if not model_path else Path(model_path)
        self.from_disk(self.model_path)

        # **Diarization params
        self.pause_threshold = 1  # RE: collapsing diarised segments
        # **ASR params
        self.batch_size = kwargs.get("batch_size", 4)
        #         self.offset = -0.18  # calibration offset for timestamps: 180 ms

        # timestamp params
        self.vocab = self.asr_model.decoder.vocabulary
        self.vocab.append("_")
        self.decoder = CTCBeamDecoder(
            self.vocab,
            beam_width=1,
            blank_id=self.vocab.index("_"),
            log_probs_input=True,
        )
        self.time_stride = (
            1 / self.asr_model.cfg.preprocessor.window_size
        )  # duration of model timesteps
        self.TIME_PAD = 1
        # huge possible max audio if model is Quartznet; maximise where possible to limit segmentation transcription error
        self.second_max_audio = (
            120
            if self.default_asr_model == "QuartzNet15x5Base-En"
            else kwargs.get("second_max_audio", 4)
        )
        # **Formatting params
        self.round_value = 3

    def from_disk(self, model_path):
        # for quicker reloading/prototyping
        #         self.dia_model = dia_model
        #         self.asr_model = asr_model
        #         self.punct_model = punct_model

        # try and load diarization model, use default otherwise
        if model_path and (model_path / "dia_model/model.pyannote").exists():
            try:
                # urgh, whatever
                with open((model_path / "dia_model/model.pyannote"), "rb") as f:
                    self.dia_model = pickle.load(f)
            except:
                asr_logger.warning(
                    f"Unable to load: {str(model_path)}, using default pyannoate instead"
                )
                self.dia_model = torch.hub.load(self.default_dia_model, "dia")
        else:
            self.dia_model = torch.hub.load(self.default_dia_model, "dia")

        # similarly for ASR model
        if model_path and (model_path / "asr_model").exists():
            try:
                self.asr_model = nemo_asr.models.ASRModel.restore_from(
                    str(model_path / "asr_model/model.nemo")
                )
            except:
                asr_logger.warning(
                    f"Unable to load: {model_path}, using default pyannoate instead"
                )
                self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.default_asr_model
                )
        else:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.default_asr_model
            )

        # similarly for punctuation
        if model_path and (model_path / "punct_model").exists():
            try:
                self.punct_model = nemo_asr.models.ASRModel.restore_from(
                    str(model_path / "punct_model/model.nemo")
                )
            except:
                asr_logger.warning(
                    f"Unable to load: {model_path}, using default pyannoate instead"
                )
                self.punct_model = PunctuationCapitalizationModel.from_pretrained(
                    self.default_punct_model
                )
        else:
            self.punct_model = PunctuationCapitalizationModel.from_pretrained(
                self.default_punct_model
            )

    def _resample_normalize_audio(self, in_file, out_file, sample_rate=16000):
        # upsample/normalize audio to 16khz WAV
        # via https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/CTC_Segmentation_Tutorial.ipynb
        if not os.path.exists(in_file):
            raise ValueError(f"{in_file} not found")
        if out_file is None:
            out_file = in_file.replace(
                os.path.splitext(in_file)[-1], f"_{sample_rate}.wav"
            )

        os.system(
            f"ffmpeg -i {in_file} -acodec pcm_s16le -ac 1 -af aresample=resampler=soxr -ar {sample_rate} {out_file} -y"
        )
        return out_file

    def _split_stereo_audio(self, in_file, out_dir):
        # into left/right channel wavs
        in_file = Path(in_file)
        out_dir = Path(out_dir)
        assert in_file.exists()
        assert out_dir.exists()

        # format output files
        left_channel = out_dir / f"{Path(in_file).stem}_left.wav"
        right_channel = out_dir / f"{Path(in_file).stem}_right.wav"

        # split, export
        audio_segment = AudioSegment.from_file(in_file)
        monos = audio_segment.split_to_mono()
        assert len(monos) == 2  # cap support at stereo audio
        monos[0].export(left_channel, format="wav")
        monos[1].export(right_channel, format="wav")

        return left_channel, right_channel

    def _diarize_mono_audio(self, in_file, single_speaker=False):
        # apply diarization to mono audio
        diarization_raw = self.dia_model({"audio": str(in_file)})
        speaker_tag_remap = {e: idx + 1 for idx, e in enumerate(string.ascii_lowercase)}
        if single_speaker:
            # split diarized segments on time delta (risky)
            # eg. in the case of stereo-split audio; assume a single speaker
            diarized_segments = (
                # format diarization as a dataframe
                pd.DataFrame(
                    [
                        {"start": turn.start, "end": turn.end, "speaker": speaker}
                        for turn, _, speaker in diarization_raw.itertracks(
                            yield_label=True
                        )
                    ]
                )
                # shift, get time deltas between diarization markers
                .assign(segment_marker=lambda x: x.start - x.end.shift(1))
                # first shifted record results in NAN; re-assign as 0 (eg. no previous record => no delta to calculate)
                .assign(segment_marker=lambda x: x.segment_marker.fillna(0.0))
                # check if deltas exceed threshold
                .assign(
                    segment_marker1=lambda x: x.segment_marker >= self.pause_threshold
                )
                # create aggregation handle
                .assign(segment_marker=lambda x: pd.Series.cumsum(x.segment_marker1))
            )
        else:
            # split diarized segments on predicted speaker attribution (less risky? - original intent of model)
            # eg. in the case of a "mashed" single audio file
            diarized_segments = (
                # format diarization as a dataframe
                pd.DataFrame(
                    [
                        {"start": turn.start, "end": turn.end, "speaker": speaker}
                        for turn, _, speaker in diarization_raw.itertracks(
                            yield_label=True
                        )
                    ]
                )
                # shift speaker attribution
                .assign(segment_marker=lambda x: x.speaker.shift(1)).assign(
                    segment_marker=lambda x: x.segment_marker != x.speaker
                )
                # create aggregation handle
                .assign(segment_marker=lambda x: pd.Series.cumsum(x.segment_marker))
            )

        diarized_segments = (
            diarized_segments
            # groupby/aggregate shifted, collapse consecutive speaker sequences
            .groupby("segment_marker")
            .agg(
                {
                    "speaker": "first",
                    "start": "first",
                    "end": "last",
                    "segment_marker": "count",
                }
            )
            .rename(
                mapper={"segment_marker": "segment_marker_count"},
                axis="columns",
                inplace=False,
            )
            .assign(segment_len=lambda x: x.end - x.start)
            # reconcile very short segments with pre/proceeding segment? merging strategy?
            #             .query('segment_len >= 1')
            # remap speaker tags from letters to numbers
            .pipe(
                lambda x: x[
                    x.speaker.apply(lambda y: True if type(y) == str else False)
                ]
            )
            .assign(
                speaker=lambda x: x.speaker.apply(
                    lambda y: int(speaker_tag_remap.get(y.lower()))
                )
            )
            .reset_index(drop=True)
        )

        return diarized_segments

    def _format_word_timestamps(self, asr_output, chunk_offset):
        preds = asr_output.y_sequence.tolist()  # some funky formatting
        probs_seq = torch.FloatTensor([preds])  # some funky formatting
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(
            probs_seq
        )
        lens = out_seq_len[0][0]
        timesteps = timesteps[0][0]

        result = []

        if len(timesteps) == 0:
            return result

        start = (timesteps[0] - self.TIME_PAD) * self.time_stride + chunk_offset
        end = (timesteps[0] + self.TIME_PAD * 2) * self.time_stride + chunk_offset

        token_prev = self.vocab[int(beam_result[0][0][0])]
        word = token_prev

        for n in range(1, lens):
            token = self.vocab[int(beam_result[0][0][n])]

            if token[0] == "#":
                # merge subwords
                word = word + token[2:]

            elif token[0] == "-" or token_prev[0] == "-":
                word = word + token

            else:
                word = word.replace("▁", "").replace("_", "")  # remove weird token

                result_word = {
                    "startTime": int(start) / 1000,
                    "endTime": int(end) / 1000,
                    "word": word,
                }
                result.append(result_word)

                start = (timesteps[n] - self.TIME_PAD) * self.time_stride + chunk_offset
                word = token

            end = (timesteps[n] + self.TIME_PAD * 2) * self.time_stride + chunk_offset
            token_prev = token

        # add last word
        word = word.replace("▁", "").replace("_", "")

        result_word = {
            "startTime": int(start) / 1000,
            "endTime": int(end) / 1000,
            "word": word,
        }
        result.append(result_word)
        return result

    def _gcp_format_single_utterance(
        self, time_formatted_words_single, channel_tag=None
    ):
        # GCP STT consistency etc. why are we standardising on this again?
        string_formatted_word_stamps = []
        for e in time_formatted_words_single:
            temp = deepcopy(e)
            temp["startTime"] = f"{e['startTime']}s"
            temp["endTime"] = f"{e['endTime']}s"
            string_formatted_word_stamps.append(temp)

        return {
            "alternatives": [
                {
                    "transcript": " ".join(
                        e["word"] for e in time_formatted_words_single
                    ),
                    "words": string_formatted_word_stamps,
                }
            ],
            "speakerTag": time_formatted_words_single[0]["speakerTag"],
            "channelTag": "None" if not channel_tag else channel_tag,
            "languageCode": "en",
        }

    def _gcp_format_aggregate_transcript(self, time_formatted_words_all):
        # GCP STT consistency etc.
        transcript_all = [
            " ".join(e["word"] for e in segment_transcript)
            for segment_transcript in time_formatted_words_all
        ]

        return " ".join(transcript_all)

    def _gcp_format_channel_seperated_transcript_objects(
        self, gcp_formatted_left_res, gcp_formatted_right_res
    ):
        # merge, sort individual left/right transcripts
        merged_utterances = pd.concat(
            [
                format_utterances_df(gcp_formatted_left_res),
                format_utterances_df(gcp_formatted_right_res),
            ]
        ).sort_values(by=["startTime", "endTime"])

        # use any/left metadata as base (should be the same file right?)
        merged_metadata = {
            k: v
            for k, v in gcp_formatted_left_res["metadata"].items()
            if k != "transcript"
        }
        merged_transcript = " ".join(merged_utterances.transcript.tolist())
        merged_metadata["transcript"] = merged_transcript

        return {
            "metadata": merged_metadata,
            "streaming_outputs": (
                merged_utterances.pipe(
                    lambda x: x[
                        ["alternatives", "speakerTag", "channelTag", "languageCode",]
                    ]
                ).to_dict(orient="records")
            ),
        }

    def _naively_segment_utterances(self, record):
        # apply naive splitting
        n_chunks = int((record.end - record.start) // self.second_max_audio) + 1
        chunk_len = (record.end - record.start) / n_chunks

        df_temp = pd.DataFrame([record] * n_chunks).reset_index(drop=True)
        df_temp["start"] = df_temp.apply(
            lambda x: x.start + chunk_len * x.name, axis=1
        )  # increase start time
        df_temp["end"] = df_temp.apply(
            lambda x: x.start + chunk_len, axis=1
        )  # increase start time
        df_temp.loc[
            (n_chunks - 1), "end"
        ] = (
            record.end
        )  # adjust end time to actual time (sanity correction in case rounding cuts of audio)
        return df_temp.assign(segment_len=lambda x: x.end - x.start)

    def _segment_utterances(self, audio_segment, record):
        dBFS = audio_segment.dBFS  # audio volume (silence level is relative to volume)
        silences = detect_silence(
            audio_segment, min_silence_len=500, silence_thresh=dBFS - 20
        )  # 0.5 break, time in ms, silence_thresh 20 lower than audio volume

        if len(silences) == 0:
            # no silence detected, lower min_silence_len
            silences = detect_silence(
                audio_segment, min_silence_len=100, silence_thresh=dBFS - 20
            )

            if len(silences) == 0:
                # if still no silences detected after lowering min_silence_len, split naively
                return self._naively_segment_utterances(record)

        silences = [
            [(s[1] - s[0]), s[0] / 1000, s[1] / 1000] for s in silences
        ]  # ms -> s

        df_temp = pd.DataFrame(record).T.reset_index(drop=True)

        # split on longest silence, in middle of silence so no info is lost
        while (len(silences) > 0) & any(df_temp.segment_len > self.second_max_audio):
            longest_silence = silences.pop(silences.index(max(silences)))
            middle_silence = record.start + (
                longest_silence[1] + (longest_silence[2] - longest_silence[1]) / 2
            )

            record_to_split = df_temp.query(
                f"start < {middle_silence} & end>{middle_silence} & segment_len > {self.second_max_audio}"
            )
            df_temp = df_temp.drop(record_to_split.index)

            split_utterances = pd.DataFrame(
                [record_to_split.iloc[0], record_to_split.iloc[0]]
            ).reset_index(drop=True)
            split_utterances.loc[0, "end"] = middle_silence
            split_utterances.loc[1, "start"] = middle_silence
            df_temp = (
                pd.concat([df_temp, split_utterances])
                .reset_index(drop=True)
                .assign(segment_len=lambda x: x.end - x.start)
            )

        if any(df_temp.segment_len > self.second_max_audio):
            # if any segments are still too long, naively split them
            final_df = [df_temp.query(f"segment_len < {self.second_max_audio}")]
            records_to_split = df_temp.query(f"segment_len > {self.second_max_audio}")

            for i, record in records_to_split.iterrows():
                final_df.append(self._naively_segment_utterances(record))
            return pd.concat(final_df).reset_index(drop=True).sort_values(by=["start"])

        return df_temp

    def _transcribe_mono(self, input_file, single_speaker=False):
        # transcribe a mono wav file
        input_file = Path(input_file)
        asr_logger.info(f"Transcribing: {input_file}..")

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1.0 resample, convert to wav
            wav_path = self._resample_normalize_audio(
                input_file, str(Path(temp_dir) / f"{Path(input_file).stem}.wav")
            )
            audio_segment = AudioSegment.from_file(wav_path)

            # 2.0 diarize input, save diarised segments
            diarized_segments = self._diarize_mono_audio(wav_path, single_speaker)
            paths2audio_files = []  # explicitly sequence, RE: sorted() issues

            chunked_diarized_segments = []
            for idx, record in diarized_segments.iterrows():
                if record.segment_len > self.second_max_audio:
                    records = self._segment_utterances(
                        audio_segment[
                            floor(record.start * 1000) : ceil(record.end * 1000)
                        ],
                        record,
                    )
                    chunked_diarized_segments.append(records)
                else:
                    chunked_diarized_segments.append(
                        pd.DataFrame(record).T.reset_index(drop=True)
                    )
            chunked_diarized_segments = pd.concat(
                chunked_diarized_segments
            ).reset_index(drop=True)

            for idx, record in chunked_diarized_segments.iterrows():
                # slice audio per utterance, round start/end to floor/ceil inclusively
                segment_audio = audio_segment[
                    floor(record.start * 1000) : ceil(record.end * 1000)
                ]

                # prevent misc output from printing
                segment_audio_res = segment_audio.export(
                    Path(temp_dir) / f"chunk_{idx}.wav", format="wav"
                )
                # collect segment audio path
                paths2audio_files.append(str(Path(temp_dir) / f"chunk_{idx}.wav"))

            # 3.0 batch transcribe, retrieve transcripts, alignments and logprobs for each utterance
            outputs = self.asr_model.transcribe(
                paths2audio_files=paths2audio_files,
                batch_size=self.batch_size,
                return_hypotheses=True,
            )

            # 4.0 retrieve/format timestamps
            time_formatted_words_all = []
            for idx, record in chunked_diarized_segments.iterrows():
                time_formatted_words = self._format_word_timestamps(
                    outputs[idx], record.start
                )

                # 5.0 apply punctuation to each output
                punctuated_sequence = self.punct_model.add_punctuation_capitalization(
                    [" ".join(e["word"] for e in time_formatted_words)]
                )[0]

                if len(punctuated_sequence.split(" ")) == len(time_formatted_words):
                    # easy case, where punctuated output len matches input len; assign directly
                    punctuated_sequence_joined = (
                        pd.DataFrame(time_formatted_words)
                        .assign(word=punctuated_sequence.split(" "))
                        .assign(speakerTag=record.speaker)
                        .to_dict(orient="records")
                    )
                    time_formatted_words_all.append(punctuated_sequence_joined)
                else:
                    # otherwise.. pad the difference? changes should be limited to immediately proceeding fullstops, commas, question marks
                    # https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html
                    print("Punctuated outputs not the same length as input")

            return {
                "metadata": {
                    "source_file": Path(input_file).name,
                    "transcript": self._gcp_format_aggregate_transcript(
                        time_formatted_words_all
                    ),
                    "duration_seconds": round(
                        audio_segment.duration_seconds, self.round_value
                    ),
                },
                "streaming_outputs": [
                    self._gcp_format_single_utterance(e)
                    for e in time_formatted_words_all
                ],
            }

    def _transcribe_channel_seperated_audio(self, input_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1.0 split left/right channels
            left_channel, right_channel = self._split_stereo_audio(input_file, temp_dir)

            # 2.0 process as seperate monos
            left_res = self._transcribe_mono(left_channel, single_speaker=True)
            right_res = self._transcribe_mono(right_channel, single_speaker=True)

        # 3.0 merge outputs
        return self._gcp_format_channel_seperated_transcript_objects(
            left_res, right_res
        )

    def predict_single(self, input_file):
        pass

    def predict_batch(self, input_files):
        return [self.predict_single(e) for e in input_files]
