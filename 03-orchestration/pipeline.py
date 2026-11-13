import pandas as pd
from prefect import flow, task
import pickle
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-orchestration")


@task(log_prints=True)
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    print(f"Loaded {len(df)} records")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Records after preparation: {len(df)} records")

    return df

@task
def create_X(df, dv=None):
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task(log_prints=True)
def train_model(X_train, y_train,dv):
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        print(f"Intercept: {lr.intercept_}")
        
        with open("preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, name="model")
        print(f"Run ID: {run.info.run_id}")
    return lr


@flow(log_prints=True)
def main_flow(year: int = 2023, month: int = 3):
    df = read_dataframe(year, month)
    X_train, dv = create_X(df)
    y_train = df['duration'].values
    lr = train_model(X_train, y_train, dv)
    return dv, lr


if __name__ == "__main__":
    main_flow()