# StreamML-classification
These are some sample files of the whole project.
The project is providing image classification as a service and now is available at https://stream.ml/.
The utils_ext is for providing the way that different paths and files can be found in the final container and paths for final outputs. In general how to connect with the external world.
The train_model_folder_trlearning_finetuning_gpu is the final code that runs when the container is called (the last line of the docker file that the container is created from is "ENTRYPOINT ["python3","/code/train_model_folder_trlearning_finetuning_gpu.py"]").
The neuralnetwork is the basic file for creating the NN.
