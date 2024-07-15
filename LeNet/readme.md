![image](https://github.com/user-attachments/assets/6db38280-3c9b-4218-833c-0777304c3c1c)

```python
python run.py
```
---
## Command-Line Arguments

–batch-size

	•	Type: int
	•	Default: 100
	•	Description: Specifies the number of samples per batch to be loaded by the data loader. Adjust this value based on the memory capacity of your GPU to optimize training performance.

–num-epochs

	•	Type: int
	•	Default: 10
	•	Description: Defines the number of complete passes through the training dataset. Increase this value to allow the model more opportunities to learn, but be mindful of overfitting.

–lr

	•	Type: float
	•	Default: 0.001
	•	Description: Sets the learning rate for the optimizer. This hyperparameter controls how much to change the model in response to the estimated error each time the model weights are updated. Lower values may result in more stable but slower convergence.

–gpu-id

	•	Type: int
	•	Default: 0
	•	Description: Indicates which GPU to use for training if multiple GPUs are available. Use the ID of the GPU as shown by your system (e.g., 0 for the first GPU).

–data-dir

	•	Type: str
	•	Default: ./dataset
	•	Description: Specifies the directory where the dataset is stored. Ensure that the data is correctly formatted and placed in this directory for successful loading and training.

–log-dir

	•	Type: str
	•	Default: ./log
	•	Description: Sets the directory where log files will be saved. These logs can be used for monitoring training progress and debugging purposes.

–save-model

	•	Type: bool
	•	Default: True
	•	Description: A flag indicating whether to save the model after training. If set to True, the trained model will be saved to the specified directory.

–eval

	•	Type: bool
	•	Default: True
	•	Description: A flag indicating whether to evaluate the model after training. If set to True, the model will be evaluated on the test dataset, and the results will be logged.

–tensorboard

	•	Type: bool
	•	Default: True
	•	Description: A flag indicating whether to enable TensorBoard logging. If set to True, training and evaluation metrics will be logged and can be visualized using TensorBoard.