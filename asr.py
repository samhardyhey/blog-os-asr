import logging
import os
import tempfile
from pathlib import Path

import nemo.collections.asr as nemo_asr
import pandas as pd
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from pyannote.audio import Pipeline
from pydub import AudioSegment, silence, utils

logging.getLogger("nemo_logger").setLevel(logging.ERROR)
ASR_LOGGER = logging.getLogger("asr")
ASR_LOGGER.setLevel(logging.INFO)

DIA_MODEL_NAME = "pyannote/speaker-diarization@2022.07"
ASR_MODEL_NAME = "stt_en_quartznet15x5"
PUNCT_MODEL_NAME = "punctuation_en_bert"

DIA_MODEL = Pipeline.from_pretrained(DIA_MODEL_NAME)
ASR_MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
PUNCT_MODEL = PunctuationCapitalizationModel.from_pretrained(PUNCT_MODEL_NAME)

PAUSE_THRESHOLD = 1
MS = 1000
BATCH_SIZE = 4
SECOND_MAX_AUDIO = 240


def resample_normalise_audio(in_file, out_file, sample_rate=16000):
    if not os.path.exists(in_file):
        raise ValueError(f"{in_file} not found")
    if out_file is None:
        out_file = in_file.replace(os.path.splitext(in_file)[-1], f"_{sample_rate}.wav")

    os.system(
        f"ffmpeg -i {in_file} -acodec pcm_s16le -ac 1 -af aresample=resampler=soxr -ar {sample_rate} {out_file} -y"
    )
    return out_file


def diarize_mono_audio(in_file, audio_segment):
    diarization_raw = DIA_MODEL(str(in_file))
    diarized_segments = (
        pd.DataFrame(
            [
                {"start": turn.start, "end": turn.end, "speaker": speaker}
                for turn, _, speaker in diarization_raw.itertracks(yield_label=True)
            ]
        )
        # shift speaker attribution > mark/collapse consecutive speaker segments
        .assign(segment_marker=lambda x: x.speaker.shift(1))
        .assign(segment_marker=lambda x: x.segment_marker != x.speaker)
        .assign(segment_marker=lambda x: pd.Series.cumsum(x.segment_marker))
        # groupby segment, merge audio start/end times
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
        # TODO: finesse a merging strategy
        .query("segment_len >= @PAUSE_THRESHOLD")
        .reset_index(drop=True)
        .assign(
            audio_segment=lambda x: x.apply(
                lambda y: _assign_child_segment(y, audio_segment), axis=1
            )
        )
    )
    return diarized_segments


def _assign_child_segment(record, parent_audio_segment):
    return parent_audio_segment[record.start * MS : record.end * MS]


def _pseudo_optimise_silence_split(audio_segment):
    # note, silence splitting has effect of reducing broader segment > small amounts of total drift
    dbfs_min = 10
    dbfs_max = 40
    dbfs_delta = 10
    min_silence_len = 500  # ms
    dBFS = audio_segment.dBFS
    audio_segments = silence.split_on_silence(
        audio_segment, min_silence_len=min_silence_len, silence_thresh=dBFS - dbfs_min
    )
    while (
        pd.Series([e.duration_seconds for e in audio_segments]).median()
        >= SECOND_MAX_AUDIO
        and dbfs_min <= dbfs_max
    ):
        ASR_LOGGER.warning(
            f"Unable to split segment on silences with silence_thresh of {dBFS - dbfs_min}; re-attempting.."
        )
        dbfs_min += dbfs_delta
        audio_segments = silence.split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=dBFS - dbfs_min,
        )

    return audio_segments


def segment_utterances(audio_segment_record):
    if audio_segment_record.segment_len > SECOND_MAX_AUDIO:
        silence_splits = _pseudo_optimise_silence_split(
            audio_segment_record.audio_segment
        )
        all_splits = []
        for split in silence_splits:
            if split.duration_seconds > SECOND_MAX_AUDIO:
                all_splits.extend(utils.make_chunks(split, SECOND_MAX_AUDIO * MS))
            else:
                all_splits.append(split)

        start_times = []
        start_time = audio_segment_record.start
        # no cumsum unfortunately, grr
        for e in all_splits:
            start_times.append(start_time)
            start_time += e.duration_seconds

        segments = (
            pd.DataFrame(
                [
                    {
                        "audio_segment": e,
                        "speaker": audio_segment_record.speaker,
                        "segment_len": e.duration_seconds,
                    }
                    for e in all_splits
                ]
            )
            .assign(start=start_times)
            .assign(end=lambda x: x.start + x.segment_len)
        )
        return segments
    else:
        return audio_segment_record.to_frame().T


def _punctuate_collapse_segment(record):
    return {
        "speaker": record.iloc[0].speaker,
        "start": record.start.min(),
        "end": record.end.max(),
        "transcript": PUNCT_MODEL.add_punctuation_capitalization(
            [" ".join(record.asr_outputs.apply(lambda x: x.text).tolist())]
        )[0],
    }


def transcribe_mono_audio(input_file, single_speaker=False):
    # transcribe a mono wav file
    input_file = Path(input_file)
    ASR_LOGGER.info(f"Transcribing: {input_file}..")

    # temp_dir = Path("../output/temp_dir")
    # shutil.rmtree(str(temp_dir)) if temp_dir.exists() else None
    # temp_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1.0 resample, convert to wav
        wav_path = resample_normalise_audio(
            input_file, str(Path(temp_dir) / f"{Path(input_file).stem}.wav")
        )
        audio_segment = AudioSegment.from_file(wav_path)
        ASR_LOGGER.info("Successfully resampled/converted input to WAV")

        # 2.0 diarize input
        diarized_segments = diarize_mono_audio(wav_path, audio_segment)
        ASR_LOGGER.info("Successfully diarized input")

        # 3.0 further segment into model-appropriate sizes
        chunked_diarized_segments = diarized_segments.apply(
            lambda x: segment_utterances(x), axis=1
        )
        chunked_diarized_segments = pd.concat(
            chunked_diarized_segments.tolist()
        ).reset_index(drop=True)
        ASR_LOGGER.info("Successfully chunked diarized input")

        paths2audio_files = []  # explicitly sequence, RE: sorted() issues
        for idx, record in chunked_diarized_segments.iterrows():
            segment_audio_res = record.audio_segment.export(
                Path(temp_dir) / f"chunk_{idx}.wav", format="wav"
            )
            paths2audio_files.append(str(Path(temp_dir) / f"chunk_{idx}.wav"))
        ASR_LOGGER.info("Successfully saved diarized segments")

        # 3.0 batch transcribe, retrieve transcripts, alignments and logprobs for each utterance
        asr_outputs = ASR_MODEL.transcribe(
            paths2audio_files=paths2audio_files,
            batch_size=BATCH_SIZE,
            return_hypotheses=True,
        )
        chunked_diarized_segments = chunked_diarized_segments.assign(
            asr_outputs=asr_outputs
        )
        ASR_LOGGER.info("Successfully processed segments with ASR model")

        return pd.DataFrame(
            chunked_diarized_segments.assign(
                segment_marker=lambda x: x.speaker.shift(1)
            )
            .assign(segment_marker=lambda x: x.segment_marker != x.speaker)
            .assign(segment_marker=lambda x: pd.Series.cumsum(x.segment_marker))
            .groupby("segment_marker")
            .apply(_punctuate_collapse_segment)
            .tolist()
        )
