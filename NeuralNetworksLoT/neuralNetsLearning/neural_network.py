import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities import binary_to_bitlist


class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.ln1 = nn.Linear(input_size, 16)
        self.ln2 = nn.Linear(16, 16)
        self.ln3 = nn.Linear(16, 1)

    def forward(self, x):
        x = 2*x-1
        x = torch.Tensor(x)
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return torch.sigmoid(x)


class CategoryLearner:

    def __init__(self,category_int,objects,batch_size,num_epochs,reps):
        """
        Parameters
        ----------
        category: category object
            A category models a function from lines of a truth table to 
            truth values. Each line of the truth table is effectively
            an object (or a set of indistinguishable objects).
            Effectively, a category is a list of bits. One for each
            possible object.
            (In effect, it can be encoded as a single int)
        """
        self.objects = objects
        # each category is a binary string of length
        # 2**n_properties
        self.category_int = category_int
        self.category = np.array(
            binary_to_bitlist(category_int,len(objects))
        ).reshape(-1,1)
        self.learning_curves = []
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reps = reps

    def train_on_category(self):
        """
        This function trains the neural network self.nn
        on 'category'. 
        
        Parameters
        ----------
        reps: int
            Number of times that the category should be learned
            (each category produces one learning curve
        """
        for rep in range(self.reps):

            learning_curve = []
            
            nn = Net(self.objects.shape[1])   
            optim = torch.optim.Adam(nn.parameters())
            num_batches = len(self.objects) // self.batch_size
            print(num_batches)

            for epoch in range(self.num_epochs):
                
                # change the order of the inputs/outputs
                permutation = np.random.permutation(len(self.objects))
                train_in = self.objects[permutation]
                train_out = self.category[permutation]

                batches_losses = []
                for batch_n in range(num_batches):

                    optim.zero_grad()

                    batch_slice = slice(
                        batch_n*self.batch_size,
                        (batch_n+1)*self.batch_size
                    )

                    # shape (batch_size, # objects)
                    batch_in = train_in[batch_slice]
                    batch_pred = nn(batch_in)

                    # shape (batch_size)
                    batch_out = train_out[batch_slice]

                    loss = F.binary_cross_entropy(
                        batch_pred,
                        torch.Tensor(batch_out)
                    )
                    batches_losses.append(
                        loss
                        .detach()
                        .numpy()
                        .item()
                    )

                    loss.backward()
                    optim.step()

                learning_curve.append(batches_losses)
            self.learning_curves.append(np.array(learning_curve))

    def save_in_database(self, con, cur):

        learning_curves = np.array(self.learning_curves)
        rep, epoch, batch = np.indices(learning_curves.shape)
        arguments = list(zip(
            rep.flatten().tolist(),
            epoch.flatten().tolist(),
            batch.flatten().tolist(),
            learning_curves.flatten().tolist()
        ))

        command = (
            'INSERT INTO data (category, rep, epoch, batch, loss) '
            f'VALUES ({self.category_int},?,?,?,?)'
        )

        cur.executemany(
            command,
            arguments
        )
