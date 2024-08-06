from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle


class Sarc_Ur5e(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0') #change the version if you want to build another dataset, if not it will overwrite the current one
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({

                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),

                        'hand_image': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.'
                        ),

                        'robot_state': tfds.features.Tensor(
                            shape=(15,),
                            dtype=np.float64,
                            doc='joint0, joint1, joint2, joint3, joint4, joint5, x,y,z, qx,qy,qz,qw, gripper_is_closed, action_blocked.'
                        ),

                        
                        'natural_language_instruction': tfds.features.Text(
                            doc='Language Instruction.'
                         ),

                        'natural_language_embedding': tfds.features.Tensor(
                            shape=(512,),
                            dtype=np.float32,
                            doc='Kona language embedding. '
                                'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                        ),
                                            
                    }),


                    'action': tfds.features.FeaturesDict({
                        'world_vector':tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='delta change in x,y,z'
                        ),

                        'rotation_delta':tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='delta change in roll, pitch, yaw'
                        ),

                        'gripper_closedness_action':tfds.features.Tensor(
                            shape=(),
                            dtype=np.float32,
                            doc='1 if close gripper, -1 if open gripper, 0 if no change.'
                        ),

                        'terminate_episode':tfds.features.Tensor(
                            shape=(),
                            dtype=np.float32,
                            doc=''
                        ),
                    }),
                        
                    
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),

                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
            
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
           
            
            data = np.load(episode_path, allow_pickle=True)

            
            # Assemble episode
            episode = []

            # Determine the length of the arrays (assuming all arrays are the same length)
            length = len(next(iter(data.values())))

            for i in range(length):
                # Extract step values for index i
                step = {}
                
                for key in data:
                    if isinstance(data[key], dict):
                        # Access nested values
                        for nested_key in data[key]:
                            if isinstance(data[key][nested_key], np.ndarray):
                                step[nested_key] = data[key][nested_key][i]
                            else:
                                step[nested_key] = data[key][nested_key]
                    else:
                        step[key] = data[key][i] if isinstance(data[key], np.ndarray) else data[key]
                # Prepare task
                task_array = step.get('task')
                task_list = task_array.flatten().tolist() if isinstance(task_array, np.ndarray) else []
                step_data = {
                    'language_instruction': task_list[0]
                }

                language_embedding = self._embed([step_data['language_instruction']])[0].numpy()
                
                if 'action' in step:
                    action_data = np.array(step['action']).astype(np.float32)


                episode.append({
                    'observation': {
                        'image': step.get('image'),
                        'robot_state': step.get('robot_state'),
                        'hand_image': step.get('hand_image'),  
                        'natural_language_instruction': step_data['language_instruction'],
                        'natural_language_embedding': language_embedding,      
                    },
                    'action': {
                        'world_vector': action_data[:3],
                        'rotation_delta': action_data[3:6],
                        'gripper_closedness_action': action_data[5],
                        'terminate_episode': action_data[-1] 
                    },
                    
                    'discount': 1.0,
                    'reward': float(i == (length - 1)),
                    'is_first': i == 0,
                    'is_last': i == (length - 1),
                    'is_terminal': i == (length - 1),

                })

            # Create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return episode_path, sample

        
        # create list of all examples
        path='/home/user/rlds_dataset_builder/sarc_ur5e/data/traj*/standard_output.pkl' #path to your dataset
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
           yield _parse_example(sample)

        #for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        #beam = tfds.core.lazy_imports.apache_beam
        #return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        #)


