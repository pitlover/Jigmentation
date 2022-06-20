import wget
import os


def my_app() -> None:
    pytorch_data_dir = "/data/Datasets"
    dataset_names = [
        # "potsdam",
        "cityscapes"]
        # "cocostuff"]
    # "potsdamraw"]
    url_base = "https://marhamilresearch4.blob.core.windows.net/stego-public/pytorch_data/"

    os.makedirs(pytorch_data_dir, exist_ok=True)
    for dataset_name in dataset_names:
        if (not os.path.exists(os.path.join(pytorch_data_dir, dataset_name))) or \
                (not os.path.exists(os.path.join(pytorch_data_dir, dataset_name + ".zip"))):
            print("\n Downloading {}".format(dataset_name))
            wget.download(url_base + dataset_name + ".zip", os.path.join(pytorch_data_dir, dataset_name + ".zip"))
        else:
            print("\n Found {}, skipping download".format(dataset_name))


if __name__ == "__main__":
    my_app()
