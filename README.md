# RLDS Dataset Conversion

Follow the guide [Installation](https://github.com/kpertsch/rlds_dataset_builder/tree/main?tab=readme-ov-file#installation) and [Run Example RLDS Dataset Creation](https://github.com/kpertsch/rlds_dataset_builder/tree/main?tab=readme-ov-file#run-example-rlds-dataset-creation).

## To build a new dataset

1. Open ```sarc_ur5e/sarc_ur5e/sarc_ur5e_dataset_builder.py``` .Modify the version so it does not overwrite the first version of dataset
    ```
    VERSION = tfds.core.Version('1.0.0') 
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
   ```
2. Make sure the ```path``` is directed to your raw dataset
3. Go to terminal. Inside the dataset directory, run:
   
   ```
   tfds build --overwrite
   ```
   
5. A new version of dataset will be saved under ```tensorflow_datasets``` folder in your local directory.

## Visualize Converted Dataset

1. Open ```visualize_dataset.ipynb```. Make sure the name assigned to ```dataset_name``` is correct. Run the cell.
   
