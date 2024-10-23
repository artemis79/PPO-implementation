import h5py
import hashlib
import os
import csv


class Logger:
    def __init__(self, args_string):
        hash_params = self._get_hash_params(args_string)
        directory = 'all_runs/' + 'run_' + hash_params
        if not os.path.exists(directory):
            os.mkdir(directory)
        self._log_hyperparameters(directory, args_string)

        self.file_observation = h5py.File(directory + '/observations.h5', 'w')
        self.file_rewards = h5py.File(directory + '/rewards.h5', 'w')

        self.file_returns = open(directory + '/returns.csv', 'w', newline='')
        self.return_writer = csv.writer(self.file_returns)
    
    def log_observation(self, b_obs, update):
        self.file_observation.create_dataset(str(update), data=b_obs)

    def log_rewards(self, b_rewards, update):
        self.file_rewards.create_dataset(str(update), data=b_rewards)

    def log_episode_return(self, returns):
        self.return_writer.writerow(returns)


    def _log_hyperparameters(self, hash_params, str_params):
        self.file_params = open(hash_params + "/hyperparameters.txt", 'w')
        self.file_params.write(str_params)
        self.file_params.close()
        

    def done(self):
        self.file_observation.close()
        self.file_returns.close()


    def _get_hash_params(self, params_str):
        hash_object = hashlib.sha256(params_str.encode())
        return hash_object.hexdigest()

