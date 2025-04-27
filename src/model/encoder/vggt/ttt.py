import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import copy

class TTT_MLP(nn.Module):
    def __init__(
        self,
        embed_dim = 2048,
    ):
        super().__init__()
        # TODO: We have global and frame features, divide them
        ## we add ttt_layer in each block to remain it to train on inference
        self.embed_dim = embed_dim

        self.ttt_mlp = nn.Sequential(
                            nn.Linear(self.embed_dim, self.embed_dim*4),
                            nn.GELU(),
                            # nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity(),
                            nn.Linear(self.embed_dim*4, self.embed_dim),
                            # nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity(),
                            nn.LayerNorm(self.embed_dim)
                        )
        self.alpha = nn.Parameter(torch.full((1, self.embed_dim), 0.01))

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x + torch.tanh(self.alpha) * self.ttt_mlp(x)
        return x

class TTT_Layer(nn.Module):
    def __init__(
        self,
        d1,
        d2,
    ):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.task = Task(d1,d2)
        self.ttt_model = TTT_MLP(d1)
        # import pdb; pdb.set_trace()

    def update_model(self, model, grads, lr=1e-3):
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            for p, g in zip(model.parameters(), grads):
                if p.requires_grad and g is not None:
                    p = p - lr * g

    def forward(self, in_seq):
        # import pdb; pdb.set_trace()
        # state = Learner(self.task, learner_model)
        all_batch_tok = []
        B, V, _, D = in_seq.shape
        for i in range(B):
            out_seq = []
            learner_model = copy.deepcopy(self.ttt_model)
            for tok in in_seq[i,...]:
                # state.train(tok)
                # out_seq.append(state.predict(tok))
                proxy_loss = self.task.loss(learner_model, tok)
                grads = torch.autograd.grad(proxy_loss, learner_model.parameters(), create_graph=True)
                self.update_model(learner_model, grads)
                predict_tok = learner_model(tok @ self.task.theta_Q)
                out_seq.append(predict_tok)
            batch_tok = torch.stack(out_seq).view(V, -1, D)
            all_batch_tok.append(batch_tok)
        out_tokens = torch.stack(all_batch_tok).view(B, V, -1, D)
        return out_tokens

class Task(nn.Module):
    def __init__(
            self,
            d1=2048,
            d2=2048,
        ):
        super().__init__()
        self.theta_K = nn.Parameter(torch.randn(d1, d2))
        self.theta_V = nn.Parameter(torch.randn(d1, d2))
        self.theta_Q = nn.Parameter(torch.randn(d1, d2))
    def loss(self, f, x):
        train_view = x @ self.theta_K
        label_view = x @ self.theta_V
        criterion = nn.MSELoss()
        return criterion(f(train_view), label_view)

# class OGD:
#     def __init__(self, lr=1e-3):
#         self.lr = lr

#     def step(self, model, grads):
#         with torch.no_grad():
#             for p, g in zip(model.parameters(), grads):
#                 if p.requires_grad and g is not None:
#                     p -= self.lr * g

# class Learner():
#     def __init__(
#             self, 
#             task,
#             model,
#         ):
#         super().__init__()
#         self.task = task
#         # Linear here, but can be any model
#         self.model = model # TTT()
#         # online GD here for simplicity
#         self.optim = OGD()

#     def train(self, x):
#         loss = self.task.loss(self.model, x)
#         import pdb; pdb.set_trace()
#         # grad_fn = grad(self.task.loss)
#         # grad_in = grad_fn(self.model, x)
#         grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
#         self.optim.step(self.model, grads)
#         # # grad function wrt first arg
#         # # of loss, which is self.model
#         # grad_fn = grad(self.task.loss)
#         # # calculate inner-loop grad
#         # grad_in = grad_fn(self.model, x)
#         # # starting from current params,
#         # # step in direction of grad_in,
#         # self.optim.step(self.model, grad_in)

#     def predict(self, x):
#         import pdb; pdb.set_trace()
#         test_view = x @ self.task.theta_Q
#         return self.model(test_view)

if __name__ == "__main__":
    B, V, P, D = 8, 4, 256, 2048
    d1, d2 = 2048, 2048
    in_seq = torch.randn(B, V, P, D)
    model = TTT_Layer(d1, d2)
    output = model(in_seq)

    print("Input shape:", in_seq.shape)      # [B, D]
    print("Output shape:", output.shape)     # [B, out_dim]
    print("Output:", output)