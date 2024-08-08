# RLDS Dataset Conversion

## Installation
```git clone``` this repo.
Follow the guide on [Installation](https://github.com/kpertsch/rlds_dataset_builder/tree/main?tab=readme-ov-file#installation)

## To build a new dataset

2. Open ```sarc_ur5e/sarc_ur5e/sarc_ur5e_dataset_builder.py``` . Modify the ```VERSION``` and ```RELEASE_NOTES``` so it does not overwrite the first version of dataset
    ```
    VERSION = tfds.core.Version('1.0.0') 
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
   ```
3. Make sure the ```path``` is directed to your raw dataset
4. Open your terminal. Make sure that you are in ```rlds-env``` environment. Inside the dataset directory, run:
   
   ```
   tfds build --overwrite
   ```
   
5. A new version of dataset will be saved under ```tensorflow_datasets``` folder in your local directory.

## Visualize Converted Dataset

1. Open ```visualize_dataset.ipynb```. Make sure the name assigned to ```dataset_name``` is correct. Run the cell.
   
## Modifications made
Record of modifications made to original repository.

### sarc_ur5e_dataset_builder.py
- [x] Renamed dataset
- [x] Modified Features
- [x] Modified Dataset Splits
- [x] Modified Data Conversion Code
- [x] Provided Dataset Description
- [x] Added License

### visualize_dataset.ipynb
- [x] Removed the parser argument
- [x] Changed the name of dataset
- [x] Modified the ```action``` and ```language_instruction``` component
