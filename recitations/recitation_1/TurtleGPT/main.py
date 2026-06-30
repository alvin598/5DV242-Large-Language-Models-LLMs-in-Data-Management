from corpus import Corpus
from model import TurtleGPT
from trainer import Trainer
import torch

usage = ['inference','pretraining','SFT'][1]
print(f"USAGE:{usage}")

set_ups = {
    'giga-turtle': {'c_window_size': 128, 'n_layer': 8, 'n_head': 8, 'd_model': 256, 'batch_size': 64},
    'big-turtle': {'c_window_size':128,'n_layer':5, 'n_head':6, 'd_model':192, 'batch_size':64},
    'michelangelo': {'c_window_size':64, 'n_layer': 3, 'n_head': 4, 'd_model': 64, 'batch_size': 64},
    'baby-turtle': {'c_window_size':32,'n_layer':1, 'n_head':1, 'd_model':32, 'batch_size':64},
    'nano-turtle': {'c_window_size':32,'n_layer':1, 'n_head':1, 'd_model':4, 'batch_size':64}
}

turtle_type = 'giga-turtle'
LEARNING_RATE = 0.0005

vocab_size = 100
corpus_file = "corpora/stories.txt"

train_dataset = Corpus(set_ups[turtle_type]['c_window_size'], corpus_file, vocab_size=vocab_size)

model = TurtleGPT(vocab_size=train_dataset.get_vocab_size(),
                  n_layer=set_ups[turtle_type]['n_layer'],
                  d_model=set_ups[turtle_type]['d_model'],
                  n_head=set_ups[turtle_type]['n_head'],
                  c_window_size=set_ups[turtle_type]['c_window_size'])


if __name__ == '__main__':

    if usage == 'inference':
        model.eval()
        with torch.no_grad():
            while True:
                prompt = input("prompt:") + " "
                x = torch.tensor(train_dataset.encode(prompt), dtype=torch.long)[None, ...].to('cuda' if torch.cuda.is_available() else 'cpu')
                y = model.generate(x, 500, temperature=0.001)[0]
                completion = ''.join([train_dataset.itot[int(i)] for i in y])

                if completion.find("$") > -1:
                    completion = completion[0:completion.find("$")]
                print('---------------------')
                print(f'    prompt: {prompt}')
                print(f"completion: {completion.replace('æ', ' ')[len(prompt):]}")
                print('---------------------')

    elif usage == 'pretraining':
        prompt_file = corpus_file.split(".txt")[0] + "-prompts.txt"
        trainer = Trainer(model,
                          train_dataset,
                          learning_rate=LEARNING_RATE,
                          prompt_file=prompt_file,
                          batch_size=set_ups[turtle_type]['batch_size'])
        trainer.run()

    elif usage == 'SFT':
        corpus_file = "corpora/stories-sft.txt"
        prompt_file = corpus_file.split(".txt")[0] + "-prompts.txt"
        cram_down = Corpus(set_ups[turtle_type]['c_window_size'],
                           corpus_file,
                           sft=True)

        cram_down.ttoi = train_dataset.ttoi
        cram_down.itot = train_dataset.itot
        cram_down.vocab_size = train_dataset.get_vocab_size()

        trainer = Trainer(model,
                          cram_down,
                          learning_rate=LEARNING_RATE,
                          prompt_file=prompt_file,
                          batch_size=set_ups[turtle_type]['batch_size'])
        trainer.run()
    else:
        print(f"undefined usage:{usage}")


