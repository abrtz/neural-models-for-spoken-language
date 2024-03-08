import os
import transformers
import numpy as np
from tqdm.auto import tqdm
import glob
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoModel, AutoProcessor
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import torch.nn as nn
import torch.optim as optim
import gc
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, geom_point, scale_x_continuous
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the TIMIT dataset
timit = load_dataset('timit_asr', data_dir='timit')

# Code for generating and saving phoneme representation (hidden state
# activations of the speech models)

def saving_phoneme_representations(model_name = "facebook/wav2vec2-base",
                                   overwrite = False,
                                   split = 'test',
                                   save_dir = '/content/drive/MyDrive/Colab Notebooks/NLP/'):

  model_savename = model_name.replace('/','-')
  savename = os.path.join(save_dir, f'{model_savename}_timit-{split}_phoneme-representations.pt')
  # Load the pre-trained wav2vec2 model and processor
  model = AutoModel.from_pretrained(model_name).to(device)
  # processor = AutoProcessor.from_pretrained(model_name)
  # Define a function to extract phoneme-level representations
  def extract_phoneme_representations(example):
    phoneme_representations, labels =[], []
    audio_input = example['audio']['array']
    phoneme_info = example["phonetic_detail"]
    phoneme_info = list(zip(phoneme_info['start'], phoneme_info['stop'], phoneme_info['utterance']))
    samples_in_frame = example['audio']['sampling_rate']*0.02
    # Tokenize the audio and get hidden states from the model
    with torch.no_grad():
        inputs = torch.from_numpy(audio_input).unsqueeze(0).to(torch.float32)
        hidden_states = torch.stack(model(inputs.to(device), output_hidden_states = True).hidden_states).detach()
        # inputs = processor(audio_input, return_tensors="pt")
        # hidden_states = torch.stack(model(**inputs.to(device), output_hidden_states = True).hidden_states).detach()

    # Extract individual vectors for each phoneme
    for start_sample, stop_sample, phoneme_ident in phoneme_info:
        start_frame, stop_frame = int(start_sample/samples_in_frame), int(stop_sample/samples_in_frame)
        stop_frame = int(start_frame + 1) if start_frame == stop_frame else stop_frame
        phoneme_vectors = torch.mean(hidden_states[:,:, start_frame:stop_frame, :], dim = -2)
        phoneme_representations.append(phoneme_vectors)
        labels.append(phoneme_ident)
    return torch.stack(phoneme_representations), np.array(labels)
  # Load the phoneme representations if they exist already
  # Generate and save them if they do not exist
  if os.path.isfile(savename) and not overwrite:
    print(f"{savename} exists already! loading it instead of generating a new one")
    phoneme_representations = torch.load(savename)
  else:
    phoneme_representations = [extract_phoneme_representations(x) for x in tqdm(timit[split])]
    torch.save(phoneme_representations,  savename)
  return phoneme_representations

# Code for processing phoneme representation (hidden state activations of the
# speech models) into input for a Classifier
def generating_classifier_input(phoneme_representations, overwrite = False):
  phoneme_vectors, labels = zip(*phoneme_representations)
  phoneme_vectors = torch.cat(phoneme_vectors).cpu()
  labels = np.concatenate(labels)

  mask_remove_silence = (labels != 'h#')

  phoneme_vectors = phoneme_vectors[mask_remove_silence]
  labels = labels[mask_remove_silence]

  classifier_input = {'input': phoneme_vectors,
                      'labels': labels}
  return classifier_input

# Code for running phoneme classification in PyTorch

# Define a simple classifier in PyTorch
class PhonemeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PhonemeClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def classification(classifier_input):
  # With the sklearn's train_test_split function, we generate 80 percent
  # training data for our classifier and 20 percent validation data
  train_features, val_features, train_labels, val_labels = train_test_split(
    classifier_input['input'].squeeze(),
    classifier_input['labels'],
    test_size=0.2,
    random_state=42)

  # Then we use a LabelEncoder to process the phoneme labels that are strings
  # into a tensor so our classifier can take them as inputs
  le = preprocessing.LabelEncoder()
  train_targets = torch.as_tensor(le.fit_transform(train_labels))
  val_targets = torch.as_tensor(le.transform(val_labels))

  layers = train_features.shape[1] # How many layers there are in the model
  num_classes = len(set(val_labels) | set(train_labels)) # How many unique phonemes exists in our dataset
  all_results = []

  # Train the classifier on different layers of the wav2vec2 model
  for layer in tqdm(range(layers), desc='Layers'):
      # Initialize the DataLoaders
      train_dataset = TensorDataset(train_features[:,layer,:], train_targets)
      train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

      # Initialize and train the classifier
      classifier = PhonemeClassifier(input_size=train_features.shape[-1], num_classes=num_classes).to(device)
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(classifier.parameters(), lr=0.005)

      # Training loop
      for epoch in range(5):  # You may need to adjust the number of epochs based on your dataset size
          for batch_features, batch_labels in train_loader:
              optimizer.zero_grad()
              outputs = classifier(batch_features.to(device)).cpu()
              loss = criterion(outputs, batch_labels)
              loss.backward()
              optimizer.step()

      # Validation
      classifier.eval()
      with torch.no_grad():
          val_predictions = torch.argmax(classifier(val_features[:,layer,:].to(device)), dim=1).detach().cpu().numpy()

      # Evaluate accuracy
      acc_score = accuracy_score(val_targets.numpy(), val_predictions)

      f1 = f1_score(val_targets.numpy(),val_predictions, average='weighted')
      results = {
          'layer': layer,
          'acc_score': acc_score,
          'f1_score': f1,
      }
      all_results.append(results)
  return all_results


# Code for Plotting the results

def plotting_results(results, model_name):
  results_dataframe = pd.DataFrame(results)
  # Create a line plot
  plot = (ggplot(results_dataframe, aes(x="layer", y="acc_score"))
  + geom_point()
  + geom_line()
  + scale_x_continuous(breaks=range(int(min(results_dataframe['layer'])), int(max(results_dataframe['layer'])) + 1))
  + labs(title=f"Phoneme classification accuracy per Layer for \n{model_name}", x="Layer", y="Accuracy")
  )
  return plot


# Defining phoneme_to_ipa dictionary
phoneme_to_ipa = {
    'aa': 'ɑ',
    'ae': 'æ',
    'ah': 'ʌ',
    'ao': 'ɔ',
    'aw': 'aʊ',
    'ax': 'ə',
    'ax-h': 'ə',
    'axr': 'ɚ',
    'ay': 'aɪ',
    'b': 'b',
    'bcl': 'b',
    'ch': 'tʃ',
    'd': 'd',
    'dcl': 'd',
    'dh': 'ð',
    'dx': 'ɾ',
    'eh': 'ɛ',
    'el': 'l̩',
    'em': 'm̩',
    'en': 'n̩',
    'eng': 'ŋ̍',
    'epi': 'ˈ',
    'er': 'ɝ',
    'ey': 'eɪ',
    'f': 'f',
    'g': 'ɡ',
    'gcl': 'ɡ',
    'h#': '#',
    'hh': 'ɦ',
    'hv': 'h',
    'ih': 'ɪ',
    'ix': 'ɨ',
    'iy': 'i',
    'jh': 'dʒ',
    'k': 'k',
    'kcl': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',
    'nx': 'ɾ̃',
    'ow': 'oʊ',
    'oy': 'ɔɪ',
    'p': 'p',
    'pau': 'p',
    'pcl': 'p',
    'q': 'ʔ',
    'r': 'ɹ',
    's': 's',
    'sh': 'ʃ',
    't': 't',
    'tcl': 't',
    'th': 'θ',
    'uh': 'ʊ',
    'uw': 'u',
    'ux': 'ʉ',
    'v': 'v',
    'w': 'w',
    'y': 'j',
    'z': 'z',
    'zh': 'ʒ'
}

def visualize_phoneme_distribution(timit_test_phoneme_representations, phoneme_to_ipa):
    """
    Print and generate a bar plot to visualize the distribution of phonemes in the TIMIT dataset based on the
    phoneme representations provided as input.

    Args:
    - timit_test_phoneme_representations (list): List of tuples containing phoneme representations from the TIMIT dataset.
    - phoneme_to_ipa (dict): Dictionary mapping phoneme labels to their corresponding IPA symbols.
    """

    # Extract phoneme labels from the phoneme representations, ignoring 'h#' phoneme
    phoneme_labels = [label for _, labels in timit_test_phoneme_representations for label in labels if label != 'h#']

    # Count occurrences of each phoneme
    phoneme_counts = Counter(phoneme_labels)

    # Normalize counts to obtain distribution
    total_phonemes = sum(phoneme_counts.values())
    phoneme_distribution = {phoneme: count / total_phonemes for phoneme, count in phoneme_counts.items()}

    # Sort phoneme distribution by frequency in descending order
    sorted_distribution = sorted(phoneme_distribution.items(), key=lambda x: x[1], reverse=True)

    # Extract phoneme labels and frequencies
    phonemes, frequencies = zip(*sorted_distribution)

    # Convert phoneme labels to IPA symbols
    ipa_phonemes = [phoneme_to_ipa.get(phoneme, phoneme) for phoneme in phonemes]

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(ipa_phonemes)), frequencies, tick_label=ipa_phonemes)
    plt.xlabel('Phonemes (IPA)')
    plt.ylabel('Frequency')
    plt.title('Phoneme Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
