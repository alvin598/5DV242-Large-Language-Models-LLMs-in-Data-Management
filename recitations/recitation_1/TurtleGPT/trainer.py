import random
import torch
from torch.utils.data.dataloader import DataLoader

GRAD_NORM_CLIP = 1.0


def robust_next_batch(data_iter,train_loader):
    try:
        return next(data_iter)
    except Exception:
        data_iter = iter(train_loader)
        return robust_next_batch(data_iter,train_loader)

class Trainer:

    def __init__(self, model, train_dataset, learning_rate=None, prompt_file=None,batch_size = None):

        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.learning_rate = learning_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        print("running on device", self.device)
        self.prompts = open(prompt_file, 'r').read().split("\n")
        self.iter_num = 0

    def run(self):
        model = self.model

        self.optimizer = model.configure_optimizers(self.learning_rate)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size
        )

        model.train() # torch will build computational graph to support gradient calculations.
        self.iter_num = 0
        data_iter = iter(train_loader)
        while True:
            batch = robust_next_batch(data_iter, train_loader)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            model.zero_grad(set_to_none=True)

            logits, self.loss = model(x, y)  # calls forward
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
            self.optimizer.step()

            if self.iter_num % 1 == 0:
                print(f"Iteration: {self.iter_num}: Loss {self.loss.item():.5f}")

            if self.iter_num % 5 == 0:
                model.eval()
                with torch.no_grad():
                    prompt = random.choice(self.prompts).strip() + " "
                    x = torch.tensor(self.train_dataset.encode(prompt), dtype=torch.long)[None, ...].to(
                        self.device)
                    y = model.generate(x, 500, temperature=0.001)[0]
                    completion = ''.join([self.train_dataset.itot[int(i)] for i in y])

                    if completion.find("$")>-1:
                        completion = completion[0:completion.find("$")]
                    print('---------------------')
                    print(f'    prompt: {prompt}')
                    print(f"completion: {completion.replace('æ',' ')[len(prompt):]}")
                    print('---------------------')

                print("saving model")
                torch.save(model.state_dict(), "model.pt" )

                model.train()

            self.iter_num += 1

