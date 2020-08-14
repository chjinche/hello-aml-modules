import shutil
import sys
from pathlib import Path

from azureml.contrib.dataset import FileHandlingOption
from azureml.data import TabularDataset
from azureml.pipeline.wrapper import dsl
from azureml.pipeline.wrapper.dsl.module import ModuleExecutor, OutputDirectory
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.image_schema import ImageSchema
from azureml.studio.core.utils.jsonutils import (dump_to_json_file, dump_to_json_lines)

_IMAGE_FOLDER_NAME = 'image'
_META_FILE_PATH = '_meta.yaml'
_TMP_DOWNLOAD_PATH = './download/'


def generate_image_list_schema(df, labeled_dataset_type):
    image_list = []
    categories = set()
    schema = None
    if labeled_dataset_type in {'MultiLabelClassification', 'MultiClassClassification'}:
        for index, row in df.iterrows():
            categories.add(row['label'])
            image_list.append({
                ImageSchema.DEFAULT_IMAGE_REF_COL: {
                    'file_name':
                    f"{_IMAGE_FOLDER_NAME}/{Path(row['image_url']).relative_to(Path(_TMP_DOWNLOAD_PATH).resolve())}",
                },
                ImageSchema.DEFAULT_CLASSIFICATION_COL: row['label'],
                ImageSchema.DEFAULT_ID_COL: index,
            })
        categories = sorted(list(categories))
        schema = ImageSchema.get_default_classification_schema([{
            'id': i,
            'name': category
        } for i, category in enumerate(categories)])
    else:
        raise NotImplementedError(f"'{labeled_dataset_type}' is not supported now.")

    return image_list, schema


def dump(image_list, schema, output_image_dir):
    meta = ImageDirectory.create_meta()
    if image_list:
        dump_to_json_lines(image_list, output_image_dir / ImageDirectory.IMAGE_LIST_FILE)

    if schema:
        dump_to_json_file(schema.to_dict(), output_image_dir / ImageDirectory._SCHEMA_FILE_PATH)
        meta.update_field('schema', ImageDirectory._SCHEMA_FILE_PATH, override=True)

    dump_to_json_file(meta.to_dict(), output_image_dir / _META_FILE_PATH)
    if image_list and schema:
        # generate samples
        image_dir = ImageDirectory.load(output_image_dir)
        samples = image_dir.get_samples()
        dump_to_json_file(samples, output_image_dir / ImageDirectory._SAMPLES_FILE_PATH)
        # update meta
        image_dir.meta.update_field('samples', ImageDirectory._SAMPLES_FILE_PATH, override=True)
        dump_to_json_file(image_dir.meta.to_dict(), output_image_dir / _META_FILE_PATH)


@dsl.module()
def convert(
        labeled_dataset: TabularDataset,
        output_image_dir: OutputDirectory(),
):
    # labeled_dataset must come from 'data labeling' project output dataset.
    df = labeled_dataset.to_pandas_dataframe(file_handling_option=FileHandlingOption.DOWNLOAD,
                                             target_path=_TMP_DOWNLOAD_PATH,
                                             overwrite_download=True)
    print(df.head())
    labeled_dataset_type = labeled_dataset.label['type']
    # move downloaded dataset to target image folder.
    output_image_dir = Path(output_image_dir)
    shutil.move(_TMP_DOWNLOAD_PATH, f'{output_image_dir}/{_IMAGE_FOLDER_NAME}')
    # generate image_list, schema.
    image_list, schema = generate_image_list_schema(df, labeled_dataset_type)
    # dump image_list, schema, meta and samples.
    dump(image_list, schema, output_image_dir)


if __name__ == '__main__':
    ModuleExecutor(convert).execute(sys.argv)
