import py_midicsv as pm
import mir_eval
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import shutil

def trim_long(onsets_x, notes_x, onsets_y, notes_y):
    # Determine the length of the shortest list
    min_length = min(len(onsets_x), len(onsets_y), len(notes_x), len(notes_y))

    # Truncate the longer lists to match the length of the shortest list
    onsets_x = onsets_x[:min_length]
    notes_x = notes_x[:min_length]
    onsets_y = onsets_y[:min_length]
    notes_y = notes_y[:min_length]

    return onsets_x, notes_x, onsets_y, notes_y

def sort_onsets_with_notes(onsets, notes):
    # Combine onsets and notes into a list of tuples (onset, note)
    onsets_notes = list(zip(onsets, notes))

    # Sort by the onset time (first element of each tuple)
    sorted_onsets_notes = sorted(onsets_notes, key=lambda x: x[0])

    # Unzip the sorted list back into two separate lists: sorted onsets and notes
    sorted_onsets, sorted_notes = zip(*sorted_onsets_notes)

    # Convert back to lists
    sorted_onsets = list(sorted_onsets)
    sorted_onsets = [int(val) for val in sorted_onsets]
    sorted_notes = list(sorted_notes)
    sorted_notes = [int(val) for val in sorted_notes]

    return np.asarray(sorted_onsets), sorted_notes


def prep_multi(notes):
    #converts a list of frequencies into the correct format for mir_eval.multipitch
    new_notes = []
    for i in range(len(notes)):
        new_notes.append(np.reshape(np.array(int(notes[i])), -1))
    new_x = np.array(new_notes)
    return new_x



def get_tempo(df):
    #finds all tempo values and uses
    tempo_index = df[df['event'] == 'Tempo'].first_valid_index()
    tempo = int(df.iloc[tempo_index]['channel'])

    ppqn_index = df[df['event'] == 'Header'].first_valid_index()
    ppqn = int(df.iloc[ppqn_index]['velocity'])
    return tempo, ppqn


def note_to_freq(note):
    return 440 * 2**((note - 69) / 12)



def read_format_csv(path):
    data = []
    max_columns = 0

    with open(path, 'r') as file:
        for line in file:
            row = line.strip().split(', ')
            max_columns = max(max_columns, len(row))
            data.append(row)

    for i in range(len(data)):
        while len(data[i]) < max_columns:
            data[i].append('None')

    df = pd.DataFrame(data)
    if max_columns == 7:
        df.columns = ['track', 'time', 'event', 'channel', 'note', 'velocity', '6']
    elif max_columns == 6:
        df.columns = ['track', 'time', 'event', 'channel', 'note', 'velocity']
    return df

def get_onsets(df):
    selected_rows = df[df['event'] == 'Note_on_c']
    times = selected_rows['time'].astype(int).to_numpy()
    tempo, ppqn = get_tempo(df)
    for i in range(len(times)):
        times[i] = (times[i] / ppqn) * (tempo / 1000000)
    return times


def get_offsets(df):
    selected_rows = df[df['event'] == 'Note_off_c']
    times = selected_rows['time'].astype(int).to_numpy()
    tempo, ppqn = get_tempo(df)
    for i in range(len(times)):
        times[i] = (times[i] / ppqn) * (tempo / 1000000)
    return times



def get_notes(df):
    selected_rows = df[df['event'] == 'Note_on_c']
    notes = selected_rows['note'].astype(int).to_numpy()
    x_list = []
    for i in range(len(notes)):
        notes[i] = mir_eval.util.midi_to_hz(notes[i])
        x = np.array(notes[i])
        x_list.append(x)
    return x_list


def note_eval_basic(notes_x, notes_y):
    acc = accuracy_score(notes_x, notes_y)
    f1 = f1_score(notes_x, notes_y, average='micro')
    precision = precision_score(notes_x, notes_y, average='micro')
    recall = recall_score(notes_x, notes_y, average='micro')
    summary = {"accuracy score":acc, "f1-score":f1, "precision score": precision, "recall": recall}
    return summary


def rename_file(filename):
    # Check if the filename ends with '.wav.mid'
    if filename.endswith('.wav.mid'):
        # Remove '.wav' and prepend 's' to the base name
        new_filename = 's' + filename[:-8] + '.mid'  # Remove the last 8 characters ('.wav.mid')
        return new_filename
    else:
        # If the file doesn't match the expected format, return the original filename
        return filename


folder_x = 'C:\\Users\\darlu\\PycharmProjects\\MIR eval\\input pred'
folder_y = 'C:\\Users\\darlu\\PycharmProjects\\MIR eval\\input truth'

f_measure_onsets = []
precision_onsets = []
recall_onsets = []
accuracy_notes = []
f1_notes = []
precision_notes = []
recall_notes = []

files_x = os.listdir(folder_x)


for filename in files_x:
    file_x_path = os.path.join(folder_x, filename)
    filename = rename_file(filename)
    print(filename)
    file_y_path = os.path.join(folder_y, filename)

    os.makedirs("tempcsvs", exist_ok=True)
    if os.path.exists(file_y_path):
        print('yes')
        csv_x = pm.midi_to_csv(file_x_path)
        csv_y = pm.midi_to_csv(file_y_path)





        with open("tempcsvs/csv_x.csv", "w") as f:
            f.writelines(csv_x)
        with open("tempcsvs/csv_y.csv", "w") as f:
            f.writelines(csv_y)

    #with open("example_converted.csv", "w") as f:
    #    f.writelines(csv_x)
    #with open("example_converted1.csv", "w") as f:
    #    f.writelines(csv_y)

        df_x = read_format_csv("tempcsvs/csv_x.csv")
        df_y = read_format_csv("tempcsvs/csv_y.csv")


        notes_x = get_notes(df_x)
        onsets_x = get_onsets(df_x)
        onsets_x, notes_x = sort_onsets_with_notes(onsets_x, notes_x)


        notes_y = get_notes(df_y)

        onsets_y = get_onsets(df_y)
        onsets_y, notes_y = sort_onsets_with_notes(onsets_y, notes_y)
        onsets_x, notes_x, onsets_y, notes_y = trim_long(onsets_x, notes_x, onsets_y, notes_y)

        acc = accuracy_score(notes_x, notes_y)
        result_notes = note_eval_basic(notes_x, notes_y)
        keys_note = list(result_notes)
        accuracy_notes.append(result_notes[keys_note[0]])
        f1_notes.append(result_notes[keys_note[1]])
        precision_notes.append(result_notes[keys_note[2]])
        recall_notes.append(result_notes[keys_note[3]])

        result_onsets = mir_eval.onset.evaluate(onsets_x, onsets_y)
        keys_onset = list(result_onsets)
        f_measure_onsets.append(result_onsets[keys_onset[0]])
        precision_onsets.append(result_onsets[keys_onset[1]])
        recall_onsets.append(result_onsets[keys_onset[2]])


    shutil.rmtree('tempcsvs')

mean_f_measure_onsets = np.mean(f_measure_onsets)
mean_precision_onsets = np.mean(precision_onsets)
mean_recall_onsets = np.mean(recall_onsets)

mean_accuracy_notes = np.mean(accuracy_notes)
mean_f1_notes = np.mean(f1_notes)
mean_precision_notes = np.mean(precision_notes)
mean_recall_notes = np.mean(recall_notes)


print("\n--- Note Onset Evaluation ---")
print(f"Mean F-Measure (Onsets): {mean_f_measure_onsets:.4f}")
print(f"Mean Precision (Onsets): {mean_precision_onsets:.4f}")
print(f"Mean Recall (Onsets): {mean_recall_onsets:.4f}")

print("\n--- Note Evaluation ---")
print(f"Mean Accuracy (Notes): {mean_accuracy_notes:.4f}")
print(f"Mean F1-Score (Notes): {mean_f1_notes:.4f}")
print(f"Mean Precision (Notes): {mean_precision_notes:.4f}")
print(f"Mean Recall (Notes): {mean_recall_notes:.4f}")



# len(onsets_x) == len(notes_x) == len(onsets_y) == len(notes_y) , "Arrays do not have the same length!"

#notes_x = prep_multi(notes_x)
#notes_y = prep_multi(notes_y)
#print(notes_x.size)
#print(notes_y.size)
#print(mir_eval.multipitch.evaluate(onsets_x, notes_x, onsets_y, notes_y))
