import logging
import re
import shutil
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from mutagen.mp3 import MP3
from tqdm import tqdm

POD_SCRAPE_LOGGER = logging.getLogger("pod_scrape_logger")
POD_SCRAPE_LOGGER.setLevel(logging.INFO)


def get_podcast_page_urls(page_url, base_url):
    res = requests.get(page_url)
    soup = BeautifulSoup(res.content, "html.parser")

    podcast_page_urls = []
    for a in soup.find_all("a", href=True):
        if "/radionational/programs" in a["href"] and len(Path(a["href"]).parts) > 3:
            podcast_page_urls.append(f"{base_url}{a['href']}")
    return podcast_page_urls


def get_podcast_mp3_link(page_soup):
    audio_elements = page_soup.find("audio")
    mp3_candidate_links = [e["src"] for e in audio_elements]

    if len(mp3_candidate_links) > 1:
        POD_SCRAPE_LOGGER.warning("More than 1 candidate mp3 URL found")
    else:
        return mp3_candidate_links[0]


def download_podcast_mp3(mp3_url, audio_dir, file_name):
    doc = requests.get(mp3_url)
    with open(audio_dir / f"{file_name}.mp3", "wb") as f:
        f.write(doc.content)


def get_podcast_transcript(page_soup):
    results = page_soup.find(id="transcript")
    return results.get_text(separator="\n")


def remove_excess_char(
    input_string,
):
    # new lines
    text = re.sub("[\n]{2,}", "\n", input_string)

    # tabs
    text = re.sub("[\t]{2,}", "\t", text)

    # carriage returns
    text = re.sub("[\r]{2,}", "\r", text)

    # vertical tabs
    text = re.sub("[\v]{2,}", "\v", text)

    # n-repetitive spaces
    for n in range(2, 10)[::-1]:
        text = text.replace(" " * n, " ")

    return text


def remove_transcript_artefacts(transcript):
    filtered = []
    for line in transcript.replace("\n:", ":\n").split("\n"):
        line = line.strip()

        # colon in initial fragment > speaker tag probably
        if ":" in line[:20]:
            line = line.split(":")[1].strip()

        # remove production audio overlay brackets/parens
        if "[" in line:

            line = re.sub("\[(.*?)\]", "", line)

        if "(" in line:
            line = re.sub("\(.*?\)", "", line)

        line = remove_excess_char(line)

        if len(line) == 0:
            continue

        if line.endswith(":"):
            # probably a speaker utterance mark
            continue

        filtered.append(line)

    return " ".join(filtered)


def prune_pairless_transcripts(audio_output_dir, transcript_output_dir):
    all_audio = {e.stem for e in audio_output_dir.glob("./*.mp3")}
    all_transcript = {e.stem for e in transcript_output_dir.glob("./*.txt")}

    intersecting_transcripts = all_audio.intersection(all_transcript)
    for file in audio_output_dir.glob("./*.mp3"):
        if file.stem not in intersecting_transcripts:
            POD_SCRAPE_LOGGER.warning(
                f"Could not find {file.name} in audio/transcript intersection; removing"
            )

            file.unlink()
    for file in transcript_output_dir.glob("./*.txt"):
        if file.stem not in intersecting_transcripts:
            POD_SCRAPE_LOGGER.warning(
                f"Could not find {file.name} in audio/transcript intersection; removing"
            )

            file.unlink()


def create_manifest(
    audio_output_dir, transcript_output_dir, podcast_min_len=5, podcast_max_len=15
):
    transcript_records = []
    for audio, transcript in zip(
        sorted(list(audio_output_dir.glob("./*.mp3"))),
        sorted(list(transcript_output_dir.glob("./*.txt"))),
    ):
        assert audio.stem == transcript.stem
        with open(transcript, "r") as f:
            transcript_text = f.read()
        transcript_records.append(
            {
                "transcript": transcript_text,
                "len_seconds": AudioSegment.from_mp3(audio).duration_seconds,
                "len_minutes": AudioSegment.from_mp3(audio).duration_seconds / 60,
                "audio_path": audio.resolve(),
                "transcript_path": transcript.resolve(),
            }
        )

    return (
        pd.DataFrame(transcript_records)
        .assign(stem=lambda x: x.audio_path.apply(lambda y: y.stem))
        .assign(
            transcript_len=lambda x: x.transcript.apply(
                lambda y: len(y.split(" ")))
        )
        .query("len_minutes >= @podcast_min_len & len_minutes <= @podcast_max_len")
        .assign(
            wpm=lambda x: x.apply(
                lambda y: y.transcript_len / y.len_minutes, axis=1)
        )
    )


def prune_transcripts_not_in_manifest(
    manifest, audio_output_dir, transcript_output_dir
):
    audio_file_names = [e.name for e in manifest.audio_path]
    transcript_file_names = [e.name for e in manifest.transcript_path]

    for file in audio_output_dir.glob("./*.mp3"):
        if file.name not in audio_file_names:
            POD_SCRAPE_LOGGER.warning(
                f"Could not find {file.name} in manifest; removing"
            )
            file.unlink()

    for file in transcript_output_dir.glob("./*.txt"):
        if file.name not in transcript_file_names:
            POD_SCRAPE_LOGGER.warning(
                f"Could not find {file.name} in manifest; removing"
            )
            file.unlink()


if __name__ == "__main__":
    # seperate output dirs for audio/transcripts
    OUTPUT_BASE_DIR = Path("./test_output")
    audio_output_dir = OUTPUT_BASE_DIR / "radio_national_podcasts/audio"
    shutil.rmtree(str(audio_output_dir)) if audio_output_dir.exists() else None
    audio_output_dir.mkdir(parents=True)

    transcript_output_dir = OUTPUT_BASE_DIR / "radio_national_podcasts/transcripts"
    shutil.rmtree(
        str(transcript_output_dir)
    ) if transcript_output_dir.exists() else None
    transcript_output_dir.mkdir(parents=True)

    # get most recent podcasts from RN website
    BASE_URL = "https://www.abc.net.au"
    PAGE_URL = "https://www.abc.net.au/radionational/transcripts/"

    podcast_page_urls = get_podcast_page_urls(PAGE_URL, BASE_URL)

    for podcast_page_url in tqdm(
        podcast_page_urls[:
                          5], desc=f"Downloading podcasts/transcripts for {PAGE_URL}"
    ):
        res = requests.get(podcast_page_url)
        soup = BeautifulSoup(res.content, "html.parser")

        mp3_link = get_podcast_mp3_link(soup)
        download_podcast_mp3(
            mp3_link, audio_output_dir, Path(podcast_page_url).parents[0].name
        )

        transcript_rough = get_podcast_transcript(soup)
        transcript_cleaned = remove_transcript_artefacts(transcript_rough)

        with open(
            transcript_output_dir /
                f"{Path(podcast_page_url).parents[0].name}.txt", "w"
        ) as f:
            f.write(transcript_cleaned)

    prune_pairless_transcripts(audio_output_dir, transcript_output_dir)
    manifest = create_manifest(audio_output_dir, transcript_output_dir)
    manifest.to_csv(audio_output_dir.parents[0] / "manifest.csv", index=False)
    prune_transcripts_not_in_manifest(
        manifest, audio_output_dir, transcript_output_dir)
