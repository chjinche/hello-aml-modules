from azureml.core import Workspace, Dataset, Experiment, RunConfiguration
from azureml.core.runconfig import CondaDependencies, DEFAULT_CPU_IMAGE

from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep


if __name__ == '__main__':
    # ws = Workspace.from_config()
    ws = Workspace.get(
        name='heta-EUS',
        subscription_id='e9b2ec51-5c94-4fa8-809a-dc1e695e4896',
        resource_group='thy-experiment'
    )
    dataset = Dataset.get_by_name(ws, name='duck_frog_20200713_042225').as_named_input('input_data')
    compute = ws.compute_targets['cpu-cluster']
    output_data = PipelineData(name='output_data', is_directory=True)
    runconfig = RunConfiguration()
    runconfig.environment.docker.base_image = DEFAULT_CPU_IMAGE
    pkgs = [
            'azureml-defaults==0.1.0.19498473',
            'azureml-contrib-dataset',
            'azureml-designer-core[image]==0.0.41',
            'azureml-pipeline-wrapper==0.1.0.19498473',
            ]
    runconfig.environment.python.conda_dependencies.set_pip_option('--extra-index-url=https://azuremlsdktestpypi.azureedge.net/CLI-SDK-Runners-Validation/19498473')
    for pkg in pkgs:
        runconfig.environment.python.conda_dependencies.add_pip_package(pkg)
    step = PythonScriptStep(
        script_name='convert_labeled_data_to_image_directory.py', name='Convert Labeled Data to Image Directory',
        arguments=['--labeled-dataset', dataset, '--output-image-dir', output_data],
        compute_target=compute,
        runconfig=runconfig,
        inputs=[dataset],
        outputs=[output_data],
    )
    pipeline = Pipeline(workspace=ws, steps=[step])
    exp = Experiment(ws, 'data_labeling').submit(pipeline)
    exp.wait_for_completion()
