import os
import logging
import time
import datetime
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from annubes.nn.custom_rnn import CustomRNN
from pathlib import Path
from annubes.tasks.task import Task
import numpy as np
import random
import sys

class Trainer:
    """General class for the training process."""
    def __init__(self,
                 task: Task,
                 net_settings: dict,
                 lr: float,
                 save_path: str,
                 training_stats_dict: dict,
                 ) -> None:
        self.task = task
        # create the network
        self.net = CustomRNN(self.task.Nin, net_settings['hidden_size'], self.task.Nout, net_settings)
        # load the model if it exists already

        # define loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr, weight_decay=0.1)
        self.save_path = save_path
        self.training_stats_dict = training_stats_dict

        # seeds
        self.numpy_seed = random.randrange(sys.maxsize)
        #print(numpy_seed)
        self.rng = np.random.default_rng(self.numpy_seed)  # seed for generating trials
        self.torch_seed = random.randrange(sys.maxsize)
        torch.manual_seed(self.torch_seed)

    def train(self,
              n_model: int,
              num_epochs: int,
              batch_size: int,
              scaling: float,
              catch_prob: float,
              file_accuracy: str,
              numpy_seed: int,
              dt_test: int,
              test_size: int,
              test_minibatch_size: int,
              threshold: float,
              target_performance: int
              ) -> None:
        for e in range(num_epochs):
            accuracy_over_choices = None
            n_analyzed_trials_total = 0
            if e == num_epochs-1:
                logging.warning(f'Model {n_model} did not convergence within {num_epochs}, \
                                final accuracy={accuracy_over_choices} over {n_analyzed_trials_total} trials')
                print(f'Model {n_model} did not convergence within {num_epochs}, \
                      final accuracy={accuracy_over_choices} over {n_analyzed_trials_total} trials')
                os.makedirs(f'{self.save_path}bad_runs/{n_model}')
                torch.save(self.net.state_dict(), f'{self.save_path}bad_runs/{n_model}/model_not_converged')
                np.save(os.path.join(self.save_path, str(n_model), 'numpy_train_seed.npy'), numpy_seed)
                np.save(os.path.join(self.save_path, str(n_model), 'torch_seed.npy'), self.torch_seed)
                with open(os.path.join(f'{self.save_path}bad_runs/{n_model}/training_stats.pkl'), 'wb') as f:
                        pickle.dump(self.training_stats_dict[n_model], f, pickle.HIGHEST_PROTOCOL)
            # generate trials
            trials = self.task.generate_trials(batch_size = batch_size, scaling = scaling, catch_prob = catch_prob)
            #print(trials['modality'])
            #reset gradient to avoid accumulation
            self.optimizer.zero_grad()
            # feed trials to network
            output, _ = self.net(torch.Tensor(trials['inputs']), self.task.tau, self.task.dt)

            # compute loss (dont penalize first 200ms of stimu)
            if self.task.fixation:
                fixation = trials['phases']['fixation']
                #Skip first 200ms of stimulus as punishment
                stimulus = trials['phases']['stimulus'][10:]
                time_steps_to_punish = np.concatenate((fixation, stimulus))
                loss = self.loss_fn(output[:, time_steps_to_punish, :],
                            torch.tensor(trials['outputs'][:, time_steps_to_punish, :]))
            else:
                loss = self.loss_fn(output[:, trials['phases']['stimulus'][10:], :],
                            torch.tensor(trials['outputs'][:, trials['phases']['stimulus'][10:], :]))
                            #!Loss calculated over all timesteps of trial, so not only end? as to also punish slow?

            # update weights
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
            self.net.set_weights()

            # check accuracy
            prediction = torch.argmax(output[:, -1, :], dim=1)#Accuracy only on final timestep of data
            nun_correct_prediction = np.sum(prediction.numpy() == trials['choice'])
            accuracy = nun_correct_prediction * 100 / batch_size
            #write training accuracy to file
            file_accuracy.write(f'Iteration {e}: ')
            file_accuracy.write(str(nun_correct_prediction * 100 / batch_size) + '\n')

            if  e >= 200 and e % 500 == 0:
                with torch.no_grad():
                    print(f'Starting validation of epoch {e}')
                    logging.info(f'Starting validation of epoch {e}')
                    print(f'Training Accuracy = {str(accuracy)}')
                    print('Loss ' + str(e) + ': ' + str(loss.item()))
                    self.task.dt = dt_test
                    test_trials = self.task.generate_trials(
                        batch_size = test_size,
                        scaling = scaling,
                        catch_prob = catch_prob,
                        numpy_seed = numpy_seed)

                    n_test_batch = int(test_size / test_minibatch_size)
                    n_test_correct = 0
                    n_analyzed_trials_total = 0
                    n_choices_made = 0
                    for j in tqdm(range(n_test_batch)):
                        start_idx = j * test_minibatch_size
                        end_idx = (j + 1) * test_minibatch_size

                        cur_batch = test_trials['inputs'][start_idx:end_idx]
                        cur_batch_choice = test_trials['choice'][start_idx:end_idx]

                        test_batch_output, _ = self.net(torch.Tensor(cur_batch), self.task.tau, dt_test)

                        output = test_batch_output.detach().numpy()
                        output_0 = output[:, test_trials['phases']['stimulus'], 0]
                        output_1 = output[:, test_trials['phases']['stimulus'], 1]
                        out_diff = output_1 - output_0

                        decision_time = np.argmax(np.abs(out_diff) > threshold, axis=1)

                        output_onset_0 = output[:, trials['phases']['stimulus'][0], 0]
                        output_onset_1 = output[:, trials['phases']['stimulus'][0], 1]
                        out_diff_onset_stimulus = output_onset_1 - output_onset_0

                        analysed_trials_valid_start = np.nonzero(np.abs(out_diff_onset_stimulus) <= threshold)[0]


                        analysed_trials_choice_made = np.nonzero(np.sum(np.abs(out_diff) > threshold, axis=1) != 0)[0]

                        analysed_trials_good_start_choice_made = np.intersect1d(
                            analysed_trials_valid_start, analysed_trials_choice_made)

                        choice = (out_diff[analysed_trials_good_start_choice_made, decision_time[analysed_trials_good_start_choice_made]] > 0).astype(np.int_)

                        n_analyzed_trials = len(analysed_trials_valid_start)
                        n_analyzed_trials_total += n_analyzed_trials
                        n_choices_made += len(choice)

                        n_test_correct += np.sum(cur_batch_choice[analysed_trials_good_start_choice_made] == choice)


                accuracy_over_choices = 100 * n_test_correct / n_analyzed_trials_total

                print('Testing accuracy:' + str(accuracy_over_choices))
                logging.info(f'Testing accuracy: {str(accuracy_over_choices)}')

                self.training_stats_dict[n_model][e] = {
                    'n_analyzed_trials': n_analyzed_trials_total,
                    'n_choices_made':n_choices_made,
                    'n_test_correct':n_test_correct,
                    'accuracy': accuracy_over_choices
                }
                print(self.training_stats_dict[n_model][e])

                if accuracy_over_choices > target_performance and n_analyzed_trials_total >= 0.9 * test_size:
                    print(f'Final number of analyzed trials: {n_analyzed_trials_total}')
                    logging.info(f'Final number of analyzed trials: {n_analyzed_trials_total}')
                    print(f'Stopped training with an accuracy of {np.round(accuracy_over_choices,2)} at epoch {e}.')
                    os.makedirs(f'{self.save_path}good_runs/{n_model}')
                    logging.info(f'Stopped training with an accuracy of \
                                 {np.round(accuracy_over_choices,2)} at epoch {e}.')
                    torch.save(self.net.state_dict(), f'{self.save_path}good_runs/{n_model}/model')
                    with open(os.path.join(f'{self.save_path}good_runs/{n_model}/training_stats.pkl'), 'wb') as f:
                        pickle.dump(self.training_stats_dict[n_model], f, pickle.HIGHEST_PROTOCOL)
                    np.save(os.path.join(self.save_path, str(n_model), 'numpy_train_seed.npy'), numpy_seed)
                    np.save(os.path.join(self.save_path, str(n_model), 'torch_seed.npy'), self.torch_seed)
                    break
                    # create directory for saving results if it does not exist
                # # save model
                #torch.save(net.state_dict(), f'{save_path}{n}/EPOCH/{e}/model')
