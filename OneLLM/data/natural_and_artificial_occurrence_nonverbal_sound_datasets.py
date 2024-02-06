import datasets
import os
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile
import json

from datasets import Value, Features, Sequence, Audio


_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_DESCRIPTION = """\
현실에 적용될 수 있는 인공 청각지능 발달에 필요한 데이터를 다양한 환경적 요인을 고려한 형태로 구축하는 것을 목적으로 함
"""

DATASET_KEY = "644"
BASE_DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"


class NaturalandArtificialOccurrenceNonverbalSoundDatasets(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = Features(
            {
                "RawDataInfo": {
                    "RawDataId": Value("string"),
                    "Copyrighter": Value("string"),
                    "SampleRate(Hz)": Value("int32"),
                    "Channel": Value("int32"),
                    "BitDepth(bit)": Value("int32"),
                    "RecordingDevice": Value("string"),
                    "BitRate(kbps)": Value("int32"),
                    "CollectionType": Value("string"),
                    "RecDateTime": Value("string"),
                    "RecDataLength(sec)": Value("int32"),
                    "Season": Value("string"),
                    "Weather": Value("string"),
                    "TimeZone": Value("string"),
                    "PlaceType": Value("string"),
                    "DistanceType": Value("string"),
                    "FileExtension": Value("string"),
                },
                "SourceDataInfo": {
                    "SourceDataId": Value("string"),
                    "FileExtension": Value("string"),
                    "NoOfClip": Value("int32"),
                    "ClipDataLength(sec)": Value("int32"),
                },
                "LabelDataInfo": {
                    "Path": Value("string"),
                    "LabelID": Value("string"),
                    "NumAnnotator": Value("int32"),
                    "Division1": Value("string"),
                    "Division2": Value("string"),
                    "Class": Value("string"),
                    "Desc": Value("string"),
                    "Type": Value("string"),
                    "NumSegmentation": Value("int32"),
                    "Segmentations": [Sequence(feature=[Value("float32")])],
                },
                "audio": Audio(44100),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=None,
        )

    def aihub_downloader(self, recv_path: Path):
        aihub_id = os.getenv("AIHUB_ID", None)
        aihub_pass = os.getenv("AIHUB_PASS", None)

        if not aihub_id:
            raise ValueError("데이터 다운을 위해선 환경변수로 AIHUB_ID가 설정되어 있어야 합니다! 환경변수로 AiHub의 ID를 입력해 주세요!")

        if not aihub_pass:
            raise ValueError(
                "데이터 다운을 위해선 환경변수로 AIHUB_PASS가 설정되어 있어야 합니다! 환경변수로 AiHub의 password를 입력해 주세요!"
            )

        header = {
            "id": aihub_id,
            "pass": aihub_pass,
        }
        param = "fileSn=all"

        obj = SmartDL(
            f"{BASE_DOWNLOAD_URL}?{param}",
            progress_bar=True,
            request_args={"headers": header},
            dest=str(recv_path),
        )
        obj.start(blocking=True)

    def unzip_data(self, tar_file: Path, unzip_dir: Path) -> list:
        with TarFile(tar_file, "r") as mytar:
            mytar.extractall(unzip_dir)
            os.remove(tar_file)

        part_glob = Path(unzip_dir).rglob("*.zip.part*")

        part_dict = dict()
        for part_path in part_glob:
            parh_stem = str(part_path.parent.joinpath(part_path.stem))

            if parh_stem not in part_dict:
                part_dict[parh_stem] = list()

            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.items():
            with open(dst_path, "wb") as byte_f:
                for part_path in sorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

        return list(unzip_dir.rglob("*.zip*"))

    def _split_generators(self, dl_manager):
        data_name = "Natural_and_artificial_occurrence_nonverbal_sound_datasets"
        cache_dir = Path(dl_manager.download_config.cache_dir)
        unzip_dir = cache_dir.joinpath(data_name)

        if not unzip_dir.exists():
            tar_file = cache_dir.joinpath(f"{data_name}.tar")
            self.aihub_downloader(tar_file)
            zip_file_path = self.unzip_data(tar_file, unzip_dir)

        train_split = [x for x in zip_file_path if "Training" in str(x)]
        valid_split = [x for x in zip_file_path if "Validation" in str(x)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_split,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_split,
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]

        breakpoint()

        label = json.loads(label_ls[0].open(label_ls[0].filelist[0]).read().decode("utf-8"))
        label = {x["IMAGE_NAME"]: x for x in label["annotations"]}
        for zip_file in source_ls:
            for file in zip_file.filelist:
                filename = file.filename.replace("/", "")
                label[filename]["IMAGE"] = zip_file.open(file.filename).read()
                image_id = label[filename]["IMAGE_ID"]

                yield (int(image_id), label[filename])
