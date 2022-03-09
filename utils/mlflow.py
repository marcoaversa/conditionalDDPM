import os

import mlflow
import mlflow
from mlflow.tracking import MlflowClient

from IPython.display import display, Markdown

def get_mlflow_model_by_name(experiment_name, run_name,
                             tracking_uri="http://mlflowroute-2587054aproject.ida.dcs.gla.ac.uk/",
                             download_model=True):
    # 0. mlflow basics
    mlflow.set_tracking_uri(tracking_uri)

    # # 1. use get_experiment_by_name to get experiment objec
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # # 2. use search_runs with experiment_id for string search query
    if os.path.isfile('cache/runs_names.pkl'):
        runs = pd.read_pickle('cache/runs_names.pkl')
        if runs['tags.mlflow.runName'][runs['tags.mlflow.runName'] == run_name].empty:
            # returns a pandas data frame where each row is a run (if several exist under that name)
            runs = fetch_runs_list_mlflow(experiment)
    else:
        # returns a pandas data frame where each row is a run (if several exist under that name)
        runs = fetch_runs_list_mlflow(experiment)

    # 3. get the selected run between all runs inside the selected experiment
    run = runs.loc[runs['tags.mlflow.runName'] == run_name]

    # 4. check if there is only a run with that name
    assert len(run) == 1, "More runs with this name"
    index_run = run.index[0]
    artifact_uri = run.loc[index_run, 'artifact_uri']

    # 5. load state_dict of your run
    state_dict = mlflow.pytorch.load_state_dict(artifact_uri)

    # 6. load model of your run
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = mlflow.pytorch.load_model(os.path.join(
    #         artifact_uri, "model"), map_location=torch.device(DEVICE))
    model = fetch_from_mlflow(os.path.join(
        artifact_uri, "model"), use_cache=True, download_model=download_model)

    return state_dict, model

def fetch_from_mlflow(uri, type='', use_cache=True, download_model=True):
    cache_loc = os.path.join('cache', uri.split('//')[1]) + '.pt'
    if use_cache and os.path.exists(cache_loc):
        print(f'loading cached model from {cache_loc} ...')
        model = torch.load(cache_loc)
    else:
        print(f'fetching model from {uri} ...')
        model = mlflow.pytorch.load_model(uri)
        os.makedirs(os.path.dirname(cache_loc), exist_ok=True)
        if download_model:
            torch.save(model, cache_loc, pickle_module=mlflow.pytorch.pickle_module)
    if type == 'processor':
        processor = model.processor
        model.processor = None
        del model   # free up memory space
        return processor
    if type == 'classifier':
        classifier = model.classifier
        model.classifier = None
        del model   # free up memory space
        return classifier
    return model


def display_mlflow_run_info(run):
    uri = mlflow.get_tracking_uri()
    experiment_id = run.info.experiment_id
    experiment_name = mlflow.get_experiment(experiment_id).name
    run_id = run.info.run_id
    run_name = run.data.tags['mlflow.runName']
    experiment_url = f'{uri}/#/experiments/{experiment_id}'
    run_url = f'{experiment_url}/runs/{run_id}'

    print(f'view results at {run_url}')
    display(Markdown(
        f"[<a href='{experiment_url}'>experiment {experiment_id} '{experiment_name}'</a>]"
        f" > "
        f"[<a href='{run_url}'>run '{run_name}' {run_id}</a>]"
    ))
    print('')
