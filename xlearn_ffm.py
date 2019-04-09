import xlearn as xl

# param:
#  0. task: binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002}

# Training task
ffm_model = xl.create_ffm()  # Use field-aware factorization machine (ffm)
ffm_model.setTrain("./xlearn_data/small_train.txt")  # Path of training data
# ffm_model.setTrain("./xlearn_data/small_train_ver1.txt")  # Path of training data


# Train model
ffm_model.setTXTModel("./xlearn_data/ffm_model.txt")  # txt model
ffm_model.fit(param, "./xlearn_data/ffm_model.out")

# ffm_model.setTXTModel("./xlearn_data/ffm_model_ver1.txt")  # txt model
# ffm_model.fit(param, "./xlearn_data/ffm_model_ver1.out")

# Testing model
ffm_model.setSigmoid()
ffm_model.setTest("./xlearn_data/small_test.txt")
# Output
ffm_model.predict("./xlearn_data/ffm_model.out", "./xlearn_data/ffm_output.txt")
