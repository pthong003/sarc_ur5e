## To input your own dataset

1. In ```Minimal_Training_Example.ipynb```, modify the dictionary ```DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS```. Add on

```
'sarc_ur5e':{ 

  'builder_dir': '/home/user/tensorflow_datasets/sarc__ur5e/1.0.0', #local directory 

  'trajectory_length': 15, 

  'step_map_fn':functools.partial(step_map_fn, 

  map_observation=map_observation, 

  map_action=berkeley_autolab_ur5_map_action) 

} 
```
2. In ```running_inference_using_RT1XTF_UR5_version.ipynb```, modify the ```builder``` directory.

    ```
   builder = tfds.builder_from_directory(builder_dir='/home/user/tensorflow_datasets/sarc__ur5e/1.0.0')
    ``` 

 


## To-do
- [x] Feed our own dataset to both ```Minimal_Training_Example``` and ```running_inference_using_RT1XTF_UR5_version.ipynb```
- [ ] Save the jax model from ```Minimal_Training_Example``` and convert it to TF SavedModel format.
