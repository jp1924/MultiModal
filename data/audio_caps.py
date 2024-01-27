# MIT License

# Copyright (c) 2019 AudioCaps authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datasets
from pathlib import Path
import pandas as pd
import os
from typing import Optional, List
from multiprocessing import Lock, Process
from tqdm import tqdm

try:
    import yt_dlp
except ImportError:
    raise ImportError("this code must need yt-dlp package, please run `pip install yt-dlp`")


_CITATION = """\
@inproceedings{audiocaps,
  title={AudioCaps: Generating Captions for Audios in The Wild},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle={NAACL-HLT},
  year={2019}
}
"""

_DESCRIPTION = """\
We explore audio captioning: generating natural language description for any kind of audio in the wild. We contribute AudioCaps, a large-scale dataset of about 46K audio clips to human-written text pairs collected via crowdsourcing on the  AudioSet dataset. The collected captions of AudioCaps are indeed faithful for audio inputs. We provide the source code of the models to explore what forms of audio representation and captioning models are effective for the audio captioning.
"""

_HOMEPAGE = "https://github.com/cdjkim/audiocaps"

_LICENSE = "MIT License"


_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/"
_URLs = {
    "train": _URL + "train.csv",
    "valid": _URL + "val.csv",
    "test": _URL + "test.csv",
}

YT_URL = "https://www.youtube.com/watch?v="
DURATION = 10


class SilentLogger(object):
    def debug(self, *args, **kwargs):
        return ""

    def warning(self, *args, **kwargs):
        return ""

    def error(self, *args, **kwargs):
        return ""


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class AudioCaps(datasets.GeneratorBasedBuilder):
    """Korean Naver movie review dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "audiocap_id": datasets.Value("int32"),
                    "youtube_id": datasets.Value("string"),
                    "start_time": datasets.Value("int32"),
                    "audio": datasets.Audio("48000"),
                    "caption": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["valid"],
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
        ]

    # copied from https://github.com/prompteus/audio-captioning/blob/main/audiocap/download_audiocaps.py
    def download(
        self,
        df: pd.DataFrame,
        audios_dir: Path,
        pid: Optional[int] = 0,
        lock: Optional[Lock] = None,
    ):
        """
        download yt videos specified in df, can be used in a multiprocess manner,
        just specify lock parameter, so logging goes safe
        """
        print(f"### Download process number {pid} started")
        for row in tqdm(list(df.iterrows()), desc=f"dw_ps: {pid}"):
            wav_file = audios_dir.joinpath(f"""{row[1]["youtube_id"]}.wav""")
            if wav_file.exists():
                continue

            youtube_id = row[1]["youtube_id"]
            start_time = row[1]["start_time"]
            end_time = start_time + DURATION

            dw_url = f"{YT_URL}{youtube_id}"

            # Download full video of whatever format
            audio_name = os.path.join(audios_dir, f"{youtube_id}")
            os.system(
                f"""yt-dlp -S "asr:48000" -x --quiet --audio-format wav --external-downloader aria2c --external-downloader-args 'ffmpeg_i:-ss {start_time} -to {end_time}' -o '{audio_name}.%(ext)s' {dw_url}"""
            )

    def df_to_n_chunks(self, df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
        chunk_size = len(df) // n + 1
        dfs = []
        for i in range(n):
            new_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            dfs.append(new_df)
        return dfs

    def _generate_examples(self, filepath, split):
        download_dir = Path(filepath).parent
        save_dir = download_dir.joinpath("AudioCaps", split)
        num_workers = os.getenv("AUDIO_CAPS_WORKER", 4)

        df = pd.read_csv(filepath)

        df_chunks = self.df_to_n_chunks(df, num_workers)
        ps_ls = list()
        for i in range(num_workers):
            process = Process(
                target=self.download,
                args=(df_chunks[i], save_dir, i),
            )
            process.start()
            ps_ls.append(process)

        for ps in ps_ls:
            ps.join()

        for id_, row in df.iterrows():
            wav_file = save_dir.joinpath(f"""{row["youtube_id"]}.wav""")
            if not wav_file.exists():
                continue
            yield id_, {
                "audiocap_id": row["audiocap_id"],
                "youtube_id": row["youtube_id"],
                "start_time": row["start_time"],
                "audio": str(wav_file),
                "caption": row["caption"],
            }
