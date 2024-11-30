import os
import matplotlib.pyplot as plt
import json
# get al .json files from ./results folder
files = os.listdir('./results')
files = [f for f in files if f.endswith('.json')]
files.sort()

def save_plot(conf,train_loss_in,train_loss,valid_loss, name):
    plt.figure()
    plt.plot(range(len(train_loss_in)),train_loss_in, label='train_loss_incorrect', c='r')
    plt.plot(range(len(train_loss)),train_loss, label='train_loss', linestyle='--', c='b')
    plt.plot(range(len(valid_loss)),valid_loss, label='valid_loss', linestyle='-.', c='g')
    plt.grid()
    plt.title(name)
    plt.legend()
    plt.savefig(f'./results/{name}.png')
    plt.close()

def read_results():
    with open('./results/results.md', 'w') as md_file:
        for i, (data) in enumerate(files):
            with open(f'./results/{data}', 'r') as f:
                data = json.load(f)
                config = data[f'config']
                train_loss_inc = data[f'train_loss_incorrect']
                train_loss = data[f'train_loss']    
                valid_loss = data[f'valid_loss']
            file_name = files[i].split('.json')[0]
            save_plot(config,train_loss_inc,train_loss,valid_loss, file_name)
            print(files[i].split('.json')[0])
            print(i)
            # wirte the configuration and the plot in a markdown file
            md_file.write(f'# Configuración {i}\n\n')
            md_file.write(f'```json\n')
            md_file.write(json.dumps(config, indent=4))
            md_file.write(f'\n```\n\n')
            md_file.write(f'![](./{file_name}.png)\n\n')
            # wirte last values of the losses
            md_file.write(f'Pérdida de entrenamiento incorrecto: {train_loss_inc[-1]}\n\n')
            md_file.write(f'Pérdida de entrenamiento: {train_loss[-1]}\n\n')
            md_file.write(f'Pérdida de validación: {valid_loss[-1]}\n\n')


read_results()