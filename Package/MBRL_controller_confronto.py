#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import rospy
import message_filters
import std_msgs
#from sensor_msgs.msg import JointState
#from std_msgs.msg import Float64
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import WrenchStamped
#from franka_core_msgs.msg import EndPointState, JointCommand, RobotState

import numpy as np
from collections import OrderedDict
from pytictoc import TicToc
#import quaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


#Importazione librerie. Presenti sia quelle necessarie per la costruzione della rete neurale che quelle per ROS


""" My code for calculating manipulability in the desired direction """
#Definizione classe per la rete neurale
class NN_model(nn.Module):
    
    def __init__(self, input_size, hidden_depth, hidden_size, output_size, print_NN=False):
        super(NN_model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.output_size = output_size
        self.print_NN = print_NN
        
        self.layers = OrderedDict()
        # first layer linear part:
        self.layers["lin" + str(1)] = nn.Linear(self.input_size, self.hidden_size)
        # first layer ReLU part:
        self.layers["relu" + str(1)] = nn.ReLU()
        
        
        # other inner layers linear part:
        for i in range(2, self.hidden_depth + 1):
            self.layers["drop"+ str(i)] = nn.Dropout(p=0.2)
            self.layers["lin" + str(i)] = nn.Linear(self.hidden_size, 
                                                    self.hidden_size)
            self.layers["relu" + str(i)] = nn.ReLU()
            
        # last layer just linear:
        self.layers["drop"+ str(i)] = nn.Dropout(p=0.1)
        self.layers["lin" + str(self.hidden_depth +1)] = nn.Linear(self.hidden_size, self.output_size)
        
        self.pipe = nn.Sequential(self.layers)
        
        if self.print_NN:
            print(self.pipe)
        
    
    def get_parameters(self):
        return self.pipe.parameters()
        
    
    def forward(self, x):
        return self.pipe(x)


class jey_MBRL():
    
    def __init__(self, NN_ensemble_, ensemble_size_, prediction_horizon_, samples_num_):
        
        self.NN_ensemble = NN_ensemble_
        self.ensemble_size = ensemble_size_
        self.prediction_horizon = prediction_horizon_
        self.samples_num = samples_num_

        Wrench_sub = message_filters.Subscriber("/franka_ee_wrench", WrenchStamped, queue_size =1,  buff_size = 2**20)
        # Subscription to a certain topic and message type
        JointPosition_sub = message_filters.Subscriber("/franka_q_pose", PoseStamped, queue_size =1, buff_size = 2**20)
        JointVelocity_sub = message_filters.Subscriber("/franka_q_velocity", PoseStamped, queue_size =1, buff_size = 2**20)

        sync = message_filters.ApproximateTimeSynchronizer([Wrench_sub, JointPosition_sub, JointVelocity_sub], queue_size=1, slop = 0.1 )
        #policy used by message_filters::sync::Synchronizer to match messages coming on a set of topics
        sync.registerCallback(self.joint_states_callback)
        #In the ROS setting a callback in most cases is a message handler. You define the message handler function and give it to subscribe.
        #You never call it yourself, but whenever a message arrives ROS will call you message handler and pass it the new message,
        #so you can deal with that.
        self.KD_pub = rospy.Publisher('/iiwa/state/KD', geometry_msgs.msg.PoseStamped, queue_size = 1)
        self.D_pub = rospy.Publisher("/D_information", std_msgs.msg.Float64, queue_size = 1)
        self.K_pub = rospy.Publisher("/K_information", std_msgs.msg.Float64, queue_size = 1)
        
        # Publishing a message on a certain topic
        self.tictoc = TicToc()
        self.num_train_data_files = 27
        self.num_test_data_files = 2
        
        # Definition of the limits
        self.K_uplim = 1000   # 5000
        self.K_lowlim = 50   # 500
        self.K_mean = (self.K_uplim + self.K_lowlim)/2

        self.D_uplim = 5    # 0.9
        self.D_lowlim = 0.1   # 0.1
        self.D_mean = (self.D_uplim + self.D_lowlim)/2

        self.u = np.array([self.K_mean, self.D_mean]) # initializing control action vector
        self.action_dim = 2

        self.action_mean = [self.K_mean, self.D_mean]
        self.action_std  = [self.K_uplim - self.K_mean, self.D_uplim - self.D_mean]
        self.action_norm = (self.action_mean, self.action_std)

        # load data
        a_file = open("Confronto_dataDict_robot_py2.pkl", "rb")
        self.training_dict = pickle.load(a_file)
        #self.training_dict = torch.load('tensors.pt')
        self.x_mean_v, self.x_std_v, self.y_mean_v, self.y_std_v = self.training_dict['xy_norm']
        self.xy_norm = (self.x_mean_v, self.x_std_v, self.y_mean_v, self.y_std_v)
        self.xn_train = self.training_dict['xn_train']
        self.yn_train_d = self.training_dict['yn_train']
        
        """
        Lista = torch.tensor([4.33437933e-01, -1.34527728e-01,  0.00000000e+00,  4.33437933e-01, -1.34527728e-01,  0.00000000e+00, -7.24582659e-02, -7.78561571e-01,
            1.73652520e-02, -2.43883667e+00, -4.57371573e-02,  1.65940576e+00, 8.01873818e-01,  5.85478123e-03, -6.78069001e-04,  8.99185710e-05, -6.20888243e-04,  6.99012569e-03, 1.35717106e-04,  8.19624265e-04 ])
    
        for j in range( 20 ):
    
            self.xy_norm[0][j] = Lista[j].item()
            self.xy_norm[2][j] = Lista[j].item()
        
        for j in range( 3 ):
    
            self.xy_norm[1][j] = 3.
            self.xy_norm[3][j] = 3.
        
        self.xy_norm[0][20] = self.action_mean [0]
        self.xy_norm[0][21] = self.action_mean [1]
        self.xy_norm[1][20] = self.action_std [0]
        self.xy_norm[1][21] = self.action_std [1]
        """
        
        print("self.xy_norm[1]", self.xy_norm[1])
         
        self.wrench_old_vector = np.array([0,0,0])  # initializing wrench_old for callback - calculating wrench_delta
        self.rate = rospy.Rate(6)  # 1.5 Hz
        self.x_record_norm = np.ones((6, 22))
        self.iter_buffer_ = 0


    # Data normalization    
    def Normalizing_data(self, x_tr, y_tr, x_te, y_te, action_norm_):
        # to normalize: (data-mean)/std
        action_mean_, action_std_ = action_norm_
        action_dim_ = len(action_mean_)
        
        x_m_ = x_tr.shape[0]
        x_n_ = x_tr.shape[1]
        
        y_m_ = y_tr.shape[0]
        y_n_ = y_tr.shape[1]
        
        state_dim_ = x_n_ - action_dim_
        
        x_train_ = np.copy(x_tr[:, 0:state_dim_])
        y_train_ = np.copy(y_tr)
        x_test_  = np.copy(x_te[:, 0:state_dim_])
        y_test_  = np.copy(y_te)
        
        actions_train_ = np.copy(x_tr[:, state_dim_:])
        actions_train_ = (actions_train_ - action_mean_)/action_std_
        
        actions_test_ = np.copy(x_te[:, state_dim_:]) 
        actions_test_ = (actions_test_ - action_mean_)/action_std_
        

        
        # we need the mean and std of the output to get the real value of predicitons
        y_mean_v_ = []
        y_std_v_  = []
        
        x_mean_v_ = []
        x_std_v_  = []
        
        # first Normalizing inputs
        for i in range(state_dim_):
            xs_   = np.copy(x_train_[:,i].reshape(x_m_,1))
            
            x_mean_ = np.mean(xs_)
            x_mean_v_.append(x_mean_)
            
            x_std_  = np.std(xs_)
            x_std_v_.append(x_std_)
            
            x_train_[:,i] = (x_train_[:,i] - x_mean_)/x_std_
            x_test_[:,i]  = (x_test_[:,i] - x_mean_)/x_std_
            
        x_mean_v_.extend(action_mean_)
        x_std_v_.extend(action_std_)
        
        x_train_ = np.append(x_train_, actions_train_, axis=1)
        x_test_  = np.append(x_test_,  actions_test_,  axis=1)
        
        # now normalizing the outputs
        for j in range(y_n_):
            ys_     = np.copy(y_train_[:,j].reshape(y_m_,1))
            
            y_mean_ = np.mean(ys_)
            y_mean_v_.append(y_mean_)
            
            y_std_  = np.std(ys_)
            y_std_v_.append(y_std_)
            
            y_train_[:,j] = (y_train_[:,j] - y_mean_)/y_std_
            y_test_[:,j] = (y_test_[:,j] - y_mean_)/y_std_
        
        xy_norm = (x_mean_v_, x_std_v_, y_mean_v_, y_std_v_)
        
        return x_train_, y_train_, x_test_, y_test_, xy_norm
    
    # Minibatches extraction form training data            
    def random_mini_batches(self, X, Y, mini_batch_size, seed):

        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (number of examples, input size)
        Y -- output data "label" vector of shape (number of examples, output size)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[0]                  # number of training examples
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        
        num_complete_minibatches = np.int(np.floor(m/mini_batch_size))
        # number of mini batches of size mini_batch_size in your partitionning
        
        for k in range(0, num_complete_minibatches):

            mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:

            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    def model_approximator_train(self, x_record, NN_model, Nh_, learning_rate):
        
        x_data_ = np.copy(x_record[0:Nh_, :])
        y_data_ = x_record[1:Nh_+1, 0:20] - x_record[0:Nh_, 0:20]
        
        
        NN_model_    = NN_model
        learning_rate_   = learning_rate
        
        
        NN_model_.to("cpu")  # putting the model into GPU (but not available now)
        cost_ = nn.MSELoss()  # Mean squared loss
        # Creates a criterion that measures the mean squared error (squared L2 norm) between each element 
        # in the input xxx and target yyy 
        optimizer_ = torch.optim.Adam(NN_model_.get_parameters(), lr =learning_rate_)
        NN_model_.train()
        
        # clear grads
        optimizer_.zero_grad()
    
        # Forward propagation
        x_ = torch.tensor(x_data_, dtype= torch.float32, device="cpu")
        y_ = torch.tensor(y_data_, dtype= torch.float32, device="cpu")
    
        y_hat_ = NN_model_.forward(x_)
        loss_ = cost_.forward(y_hat_, y_)              # calculating the loss
        loss_.backward()                              # backprop
        optimizer_.step()                             # updating the parameters
        torch.cuda.empty_cache()
               
            
        return
    
    # definition of cost function
    def cost_func_p(self, x_): 
        # shape is: (num_ensemble, N, dx)
        x_  = np.abs(x_)  
        dx_ = x_.shape[2]
        N_  = x_.shape[1]
        ense_ = x_.shape[0]

        cx_ = np.zeros((1, dx_))
        #cx_[0,0] = 3
        cx_[0,2] = 3
        # array([[3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            
        cost_ = (np.sum(np.sum( x_* cx_, axis=2).reshape(ense_, N_), axis=0)/ense_).reshape(N_) #+ (np.sum(np.sum( np.multiply(x_,x_)*cx_, axis=2).reshape(ense_, N_), axis=0)/ense_).reshape(N_)

        return cost_
    
    #definition of Cross Entropy Method
    def CEM_norm_p(self, x_initial_, action_dim_, time_horizon_, num_samples_, xy_norm_, NN_model_, num_ensembles_cem_):
        
        K_uplim_  = 1000; K_lowlim_ = 50;   K_mean_ = ( K_uplim_ +  K_lowlim_)/2
        D_uplim_  = 5;  D_lowlim_ = 0.1;   D_mean_ = ( D_uplim_ +  D_lowlim_)/2
        
        assert(x_initial_.shape[0] == 1)  # state should be a row vector (without actions)
        
        x_mean_v_, x_std_v_, y_mean_v_, y_std_v_ = xy_norm_
        
        #print("x_mean_v_", x_mean_v_)
        
        state_dim_  = x_initial_.shape[1]
        state_action_dim_ = action_dim_ + state_dim_
        smoothing_rate_ = 0.9
        iteration_      = 5
        num_elites_ = 8            # 16-32 elites are enough with 64-128 samples for: action_dim * time_horizon <= 100
        
        num_ensembles_ = num_ensembles_cem_
        
        for k in range(num_ensembles_):
            NN_model_["NN"+str(k)].to("cpu")
           
        # Initializing:
        mu_matrix_  = np.zeros((action_dim_, time_horizon_))
        std_matrix_ = np.ones((action_dim_, time_horizon_))
        
        for _ in range(iteration_):

            state_t_broadcasted_ = np.ones((num_ensembles_, num_samples_, state_dim_)) * x_initial_

            if 'action_samples_' in locals(): 
                del action_samples_

            # Draw random samples
            action_samples_ = np.random.normal(loc=mu_matrix_, scale=std_matrix_, size=(num_samples_, action_dim_, time_horizon_))
            action_samples_[action_samples_ >=  1] =  1
            action_samples_[action_samples_ <= -1] = -1

            costs_ = np.zeros(num_samples_)

            # Evaluate the trajectories and find the elites
            for t in range(time_horizon_):

                action_t_norm_ = action_samples_[:,:,t].reshape(num_samples_, action_dim_)
                action_t_broadcasted_norm_ = np.ones((num_ensembles_, num_samples_, action_dim_)) * action_t_norm_
                
                state_t_broadcasted_norm_ = (state_t_broadcasted_ - x_mean_v_[0:state_dim_])/x_std_v_[0:state_dim_]
                
                state_action_norm_ = np.append(state_t_broadcasted_norm_, action_t_broadcasted_norm_, axis=2)
                
                state_action_norm_torch_ = torch.tensor(state_action_norm_, dtype=torch.float32, device="cpu")
                state_t_broadcasted_norm_torch_ = torch.tensor(state_t_broadcasted_norm_, dtype=torch.float32, device="cpu")

                
                state_tt_norm_torch_ = NN_model_["NN0"].forward(state_action_norm_torch_[0,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[0,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )
                state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN1"].forward(state_action_norm_torch_[1,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[1,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)
                #state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN2"].forward(state_action_norm_torch_[2,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[2,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)

                #state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN3"].forward(state_action_norm_torch_[3,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[3,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)
                #state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN4"].forward(state_action_norm_torch_[4,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[4,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)

                # state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN5"].forward(state_action_norm_torch_[5,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[5,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)
                # state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN6"].forward(state_action_norm_torch_[6,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[6,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)

                # state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN7"].forward(state_action_norm_torch_[7,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[7,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)
                # state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN8"].forward(state_action_norm_torch_[8,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[8,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)
                # state_tt_norm_torch_ = torch.cat((state_tt_norm_torch_, NN_model_["NN9"].forward(state_action_norm_torch_[9,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[9,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )), dim=0)

                state_tt_norm_ = np.asarray(state_tt_norm_torch_.detach()).reshape(num_ensembles_, num_samples_, state_dim_)
                state_tt_ = state_tt_norm_*y_std_v_ + y_mean_v_

                step_cost_ = self.cost_func_p(state_tt_)
                state_t_broadcasted_ = state_tt_
                del state_action_norm_torch_; del state_t_broadcasted_norm_torch_
                torch.cuda.empty_cache()

                costs_ += step_cost_

            # NN_model_["NN0"].to("cpu")   # to reduce the memory consumption on gpu
                
            top_elites_index_ = costs_.argsort()[::1][:num_elites_]  # sorting index with min cost first

            elites_  = action_samples_[top_elites_index_,:,:].reshape(num_elites_, action_dim_, time_horizon_)

            mu_matrix_new_  = np.sum(elites_, axis=0)/num_elites_
            std_matrix_new_ = np.sqrt( np.sum( np.square(elites_ - mu_matrix_new_), axis=0)/num_elites_) 
            # mu_new should broadcast to size of elites_ then subtract and then elementwise square 

            # Update the mu_ and std_
            mu_matrix_  = smoothing_rate_*mu_matrix_new_  + (1-smoothing_rate_)*mu_matrix_
            std_matrix_ = smoothing_rate_*std_matrix_new_ + (1-smoothing_rate_)*std_matrix_
            best_action_n_seq_ = elites_[0,:,:].reshape(action_dim_, time_horizon_)
            
            action_mean_v_ = np.asarray(x_mean_v_[state_dim_:]).reshape(action_dim_,1)
            action_std_v_  = np.asarray(x_std_v_[state_dim_:]).reshape(action_dim_,1)
            best_action_seq_ = best_action_n_seq_*action_std_v_ + action_mean_v_
            
        
        return best_action_seq_



    def joint_states_callback(self, wrench, joint_position, joint_velocity):

        if (self.iter_buffer_ < 5):
            # print("Inside Callback")
            self.wrench_vector_now   = np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z])
            self.wrench_delta_vector = self.wrench_vector_now - self.wrench_old_vector
            self.wrench_old_vector   = self.wrench_vector_now
    
            # print("Wrench: ", self.wrench_vector_now)
            #print("msg.Wrench: ", np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z]))
            self.joint_position_vector = np.array([joint_position.pose.position.x, joint_position.pose.position.y, joint_position.pose.position.z,
                                                   joint_position.pose.orientation.x, joint_position.pose.orientation.y, joint_position.pose.orientation.z, joint_position.pose.orientation.w])
    
            self.joint_velocity_vector = np.array([joint_velocity.pose.position.x, joint_velocity.pose.position.y, joint_velocity.pose.position.z,
                                                   joint_velocity.pose.orientation.x, joint_velocity.pose.orientation.y, joint_velocity.pose.orientation.z, joint_velocity.pose.orientation.w])
    
    
            self.x = np.concatenate((self.wrench_vector_now, self.wrench_delta_vector, self.joint_position_vector, self.joint_velocity_vector), axis=0)
            self.x = self.x.reshape(1, len(self.x))
            #print("self.x", self.x)
    
    
            # self.tictoc.tic()
            self.best_u_sequence = self.CEM_norm_p(self.x, self.action_dim, self.prediction_horizon, self.samples_num, self.xy_norm, self.NN_ensemble, self.ensemble_size)
    
    
            Kr_Dr = geometry_msgs.msg.PoseStamped()
            
            # The optimal stiffness and damping in z direction are published
            # Initializing!
            Kr_Dr.pose.position.x = 500                               # K_x 2750
            Kr_Dr.pose.position.y = 500                               # K_y 2750
            Kr_Dr.pose.position.z = self.best_u_sequence[0,0]         # K_z
            
            Kr_Dr.pose.orientation.x = 0.5                            # D_x 0.5
            Kr_Dr.pose.orientation.y = 0.5                            # D_y 0.5
            Kr_Dr.pose.orientation.z = self.best_u_sequence[1,0]      # D_z
    
            self.KD_pub.publish(Kr_Dr)
            
            Dr_mes = std_msgs.msg.Float64
            Dr_mes = self.best_u_sequence[1,0]
            self.D_pub.publish(Dr_mes)
            
            Kr_mes = std_msgs.msg.Float64
            Kr_mes = self.best_u_sequence[0,0]
            self.K_pub.publish(Kr_mes)
            
            # self.rate.sleep()
            # elapsed_time = self.tictoc.tocvalue()
            print("KD Valuse :", [Kr_Dr.pose.position.z, Kr_Dr.pose.orientation.z])
            # print("Computation time :", elapsed_time)
            self.x_record_norm[self.iter_buffer_, 0:20] = (self.x - self.xy_norm[2])/(self.xy_norm[3])
            self.x_record_norm[self.iter_buffer_, 20:21] = (self.best_u_sequence[0,0] - self.action_mean [0])/(self.action_std [0])
            self.x_record_norm[self.iter_buffer_, 21:22] = (self.best_u_sequence[1,0] - self.action_mean [1])/(self.action_std [1])
            self.iter_buffer_ += 1
            
        if (self.iter_buffer_ % 5 == 0):
            
            self.wrench_vector_now   = np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z])
            self.wrench_delta_vector = self.wrench_vector_now - self.wrench_old_vector
            self.wrench_old_vector   = self.wrench_vector_now
    
            # print("Wrench: ", self.wrench_vector_now)
            self.joint_position_vector = np.array([joint_position.pose.position.x, joint_position.pose.position.y, joint_position.pose.position.z,
                                                   joint_position.pose.orientation.x, joint_position.pose.orientation.y, joint_position.pose.orientation.z, joint_position.pose.orientation.w])
    
            self.joint_velocity_vector = np.array([joint_velocity.pose.position.x, joint_velocity.pose.position.y, joint_velocity.pose.position.z,
                                                   joint_velocity.pose.orientation.x, joint_velocity.pose.orientation.y, joint_velocity.pose.orientation.z, joint_velocity.pose.orientation.w])
    
    
            self.x = np.concatenate((self.wrench_vector_now, self.wrench_delta_vector, self.joint_position_vector, self.joint_velocity_vector), axis=0)
            self.x = self.x.reshape(1, len(self.x))
            self.x_record_norm[self.iter_buffer_, 0:20] = (self.x - self.xy_norm[2])/(self.xy_norm[3])
            
            #print("MODEL")
            for n_rete in range(self.ensemble_size):
                self.model_approximator_train(self.x_record_norm, self.NN_ensemble["NN" + str(n_rete)], 5, learning_rate = 1e-3)
            
            self.x_record_norm = np.ones((6, 22))
            #print("self.x_record_norm", self.x_record_norm[5, : ] )
            self.iter_buffer_ = 0
    
# Main function.

    
if __name__=='__main__':
    
    print("Main function is started")
    # Node initialization
    rospy.init_node('jey_MBRL', anonymous=True)

    Device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} Device".format(Device))


    # Loading the data
    a_file = open("Confronto_dataDict_robot_py2.pkl", "rb")
    training_dict = pickle.load(a_file)
    #self.training_dict = torch.load('tensors.pt')
    x_mean_v, x_std_v, y_mean_v, y_std_v = training_dict['xy_norm']
    xn_train = training_dict['xn_train']
    yn_train_d = training_dict['yn_train']
    print("x_mean_v", x_mean_v)
    print("x_std_v", x_std_v)
    print("y_mean_v", y_mean_v)
    print("y_std_v", y_std_v)
    
    # Required data for NN models
    n_x = xn_train.shape[1]                        # input_size
    n_y = yn_train_d.shape[1]                      # output_size
    n_d = 5                                        # depth of the hidden layers
    n_h = 512                                      # size of the hidden layers
    num_ensembles = 2                              # number of NN in the ensemble
    T=7                                            # prediction horizon
    N=16                                           # number of samples
    print_loss = True
    dx = yn_train_d.shape[1]
    du = xn_train.shape[1] - yn_train_d.shape[1]
    NN_delta_norm_cem_ensemble = OrderedDict()

    # Initializing the NN models
    for i in range(num_ensembles):
        NN_delta_norm_cem_ensemble["NN" + str(i)] = NN_model(n_x, n_d, n_h, n_y, print_NN=False)

    # Loading the NN models
    for i in range(num_ensembles):
        Path = "/home/franka/andrea_ws/src/Package/NN_Paper_ensem_10" + str(i)
        NN_delta_norm_cem_ensemble["NN" + str(i)] = NN_model(n_x, n_d, n_h, n_y, print_NN=False)
        NN_delta_norm_cem_ensemble["NN" + str(i)].load_state_dict(torch.load(Path))

    for i in range(num_ensembles):
        NN_delta_norm_cem_ensemble["NN" + str(i)].to("cpu")
        NN_delta_norm_cem_ensemble["NN" + str(i)].eval()
    
    print("MBRL controller is starting")
    myController = jey_MBRL(NN_delta_norm_cem_ensemble, num_ensembles, T, N)
    rospy.spin()