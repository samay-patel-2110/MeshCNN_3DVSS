from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_test(epoch=-1):
    
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    
    preds = []
    labels = []    
    
    for i, data in enumerate(dataset):
        
        model.set_input(data)
        ncorrect, nexamples, p, l = model.test()
        writer.update_counter(ncorrect, nexamples)
        preds.append(p)
        labels.append(l)

    # Flatten predictions and labels
    preds_flat = np.concatenate(preds)
    labels_flat = np.concatenate(labels)
    
    # Compute confusion matrix
    confusion_matrix = np.zeros((opt.nclasses, opt.nclasses), dtype=int)
    for i in range(len(preds_flat)):
        confusion_matrix[labels_flat[i]][preds_flat[i]] += 1
        
    # Print accuracy
    writer.print_acc(epoch, writer.acc)
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    # Plot confusion matrix using seaborn
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # Save the plot
    plt.savefig(f'confusion_matrix_{opt.name}.png')
    plt.close()
    
    return writer.acc


if __name__ == '__main__':
    run_test()
