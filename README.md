#Neural Network Hidden markov model
>
1.first you should install the hmmlearn open sources tool.And also you should have tensorflow installed.
>
* algorithm:
>we use  gaussian hidden markov model to label the training dataset(you can specify how many hidden state you want to use) and after that for each state we do long action then plot the cumulative profit of each state.
>then we use the NN to train the labeled training data.Remember that the features used may be different.
>finally we use the trained NN model to classify the test data set.When the cumulative reward is larger than zero we decide to do long action if neural model classify the test data as the state correspongding to that reward and  vice versa.
>
2.we find that the result is profitable in the IF data. Please refer to the IF_hmm_state_predict  to see the final result.
