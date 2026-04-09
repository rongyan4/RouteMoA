import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from src.data_loader import get_dataloader
from src.model import RouterModel
from src.loss import bce_loss, mse_loss, kl_loss, focal_loss
from src.evaluate import evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.models.mf import MatrixFactorizationRouter
import numpy as np


custom_loss_fn = {
    'bce': bce_loss,
    'mse': mse_loss,
    'kl': kl_loss,
    'focal': focal_loss
}

def plot_metrics(losses, class_accuracy, total_accuracy, auc_per_class, macro_auc, save_path, label_num):
    plt.figure(figsize=(18, 8))

    # Plot training loss
    plt.subplot(351)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot test accuracy
    plt.subplot(352)
    plt.plot([acc["0_class_accuracy"] for acc in total_accuracy], label="0-Class")
    plt.plot([acc["1_class_accuracy"] for acc in total_accuracy], label="1-Class")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Total Accuracy")
    plt.legend()

    # Plot accuracy for each class
    for i in range(label_num):
        plt.subplot(3, 5, i+3)  
        plt.plot([acc[i]["0_class_accuracy"] for acc in class_accuracy], label=f"0-Class")
        plt.plot([acc[i]["1_class_accuracy"] for acc in class_accuracy], label=f"1-Class")
        plt.xlabel("Epoch")
        plt.ylabel(f"Accuracy")
        plt.title(f"Model {i} Accuracy")
        plt.legend()

    # Plot AUC-ROC for each label
    plt.subplot(3, 5, 13)
    auc_per_class_np = np.array(auc_per_class)  # Convert to numpy array
    for i in range(label_num):
        plt.plot(auc_per_class_np[:, i], label=f"Model {i} AUC", marker='o', linestyle='-', color=f'C{i}') 
    plt.xlabel("Epoch")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC per Class")
    plt.legend()

    # Plot macro-average AUC
    plt.subplot(3, 5, 14)
    plt.plot(macro_auc, label="Macro AUC", marker='o', linestyle='-', color='r')
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Macro Average AUC")
    plt.legend()

    # Save images
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_BERT)
    train_loader = get_dataloader(config.TRAIN_FILE, tokenizer, config.TRAIN_BATCH_SIZE, 512)
    test_loader = get_dataloader(config.TEST_FILE, tokenizer, config.TEST_BATCH_SIZE, 512)
    
    label_num = len(train_loader.dataset[0]["labels"])  # Determine number of models
    model = MatrixFactorizationRouter(config.PRETRAINED_BERT, label_num)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    train_class_accuracy = []  # Store class accuracy for each epoch
    train_total_accuracy = []  # Store total accuracy
    train_auc_per_class = []  # Store AUC-ROC for each class per epoch
    train_macro_auc = []  # Store macro-average AUC per epoch

    test_class_accuracy = []  # Store class accuracy for each epoch
    test_total_accuracy = []  # Store total accuracy
    test_auc_per_class = []  # Store AUC-ROC for each class per epoch
    test_macro_auc = []  # Store macro-average AUC per epoch

    model.eval()
    train_result = evaluate(model, train_loader)
    train_class_accuracy.append(train_result["class_accuracy"])
    train_total_accuracy.append(train_result["total_accuracy"])
    train_auc_per_class.append(train_result["auc_per_class"])
    train_macro_auc.append(train_result["macro_auc"])

    test_result = evaluate(model, test_loader)
    test_class_accuracy.append(test_result["class_accuracy"])
    test_total_accuracy.append(test_result["total_accuracy"])
    test_auc_per_class.append(test_result["auc_per_class"])
    test_macro_auc.append(test_result["macro_auc"])

    print(f"[Start!] Train Set 0-class Score: {train_result['total_accuracy']['0_class_accuracy']:.4f}, 1-class Score: {train_result['total_accuracy']['1_class_accuracy']:.4f}, Test Set 0-class Score: {test_result['total_accuracy']['0_class_accuracy']:.4f}, 1-class Score: {test_result['total_accuracy']['1_class_accuracy']:.4f}") 
    print(f"Train AUC: {train_result['macro_auc']:.4f}, Test AUC: {test_result['macro_auc']:.4f}")

    plot_metrics(losses, train_class_accuracy, train_total_accuracy, train_auc_per_class, train_macro_auc, config.TRAIN_PLOT_SAVE_PATH, label_num)
    plot_metrics(losses, test_class_accuracy, test_total_accuracy, test_auc_per_class, test_macro_auc, config.TEST_PLOT_SAVE_PATH, label_num)

    best_test_auc = 0
    last_test_auc = 0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        batch_idx = 0

        for batch in tqdm(train_loader, desc='Training'):
            batch_idx += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = custom_loss_fn['focal'](logits, labels)

            # print(f"Epoch {epoch + 1}/{config.EPOCHS}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()
        train_result = evaluate(model, train_loader)
        train_class_accuracy.append(train_result["class_accuracy"])
        train_total_accuracy.append(train_result["total_accuracy"])
        train_auc_per_class.append(train_result["auc_per_class"])
        train_macro_auc.append(train_result["macro_auc"])

        test_result = evaluate(model, test_loader)
        test_class_accuracy.append(test_result["class_accuracy"])
        test_total_accuracy.append(test_result["total_accuracy"])
        test_auc_per_class.append(test_result["auc_per_class"])
        test_macro_auc.append(test_result["macro_auc"])

        if test_result["macro_auc"] > best_test_auc:
            best_test_auc = test_result["macro_auc"]
            
        last_test_auc = test_result["macro_auc"]

        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}, Train set 0-class Score: {train_result['total_accuracy']['0_class_accuracy']:.4f}, 1-class Score: {train_result['total_accuracy']['1_class_accuracy']:.4f}, Test set 0-class Score: {test_result['total_accuracy']['0_class_accuracy']:.4f}, 1-class Score: {test_result['total_accuracy']['1_class_accuracy']:.4f}") 
        print(f"Train AUC: {train_result['macro_auc']:.4f}, Test AUC: {test_result['macro_auc']:.4f}")
        plot_metrics(losses, train_class_accuracy, train_total_accuracy, train_auc_per_class, train_macro_auc, config.TRAIN_PLOT_SAVE_PATH, label_num)
        plot_metrics(losses, test_class_accuracy, test_total_accuracy, test_auc_per_class, test_macro_auc, config.TEST_PLOT_SAVE_PATH, label_num)

        # Early stopping
        if avg_loss < best_loss - 0.0005:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break
    print(f"Training finished. Best test AUC is {best_test_auc}. Test AUC of the last epoch is {last_test_auc}.")
