import glob
import librosa
import torch
import pandas
import numpy

directory = glob.glob(r'C:\Users\octav\source\repos\MusicGenreClassifier\MusicGenreClassifier\Files\*.mp3')

print('')
print('Music classifier helper')
print('')

if len(directory) < 1:
    print('Notication: MP3 files were not founds')
else:
    column1 = []
    column2 = []
    seconds = 5
    rate = 22050
    lenght = seconds * rate
    id_num = 0
    point1 = 0
    num_points = 5
    paths = []
    max_lenght = max([len(x) for x in directory])
    limit_name = directory[0].rfind('\\') + 1
    for path in directory:
        print('\033[?25l\033[3J\033[H',end='')
        print('')
        print('Music classifier helper')
        print('')
        print('Loading' + '.' * point1 + ' ' * (num_points - point1))
        print(path + ' ' * (max_lenght - len(path)))
        paths.append(path)
        wave, sr = librosa.load(path, sr=22050)
        parts = max(len(wave)//(sr * seconds), 1)
        column1.extend(numpy.array_split(wave, parts))
        column2.extend(parts * [id_num])
        id_num += 1
        point1 += 1
        
        if point1 > num_points:
            point1 = 0

    frame = pandas.DataFrame()
    frame['genres'] = column1
    frame['id'] = column2

    from Dataloader import CustomAudioDataset, spectogram
    from torch.utils.data import DataLoader
    import torch
    dataset = CustomAudioDataset(frame, spectogram, rate, lenght)
    validation = DataLoader(dataset, batch_size=128, shuffle=False)

    from AudioNet import NeuralNet
    model = NeuralNet()
    model.load_state_dict(torch.load('C:\\Users\\octav\\source\\repos\\MusicGenreClassifier\\MusicGenreClassifier\\wmusic.pth'))
    model.eval()

    results = []
    point2 = 0
    with torch.no_grad():
        for data in validation:
            print('\033[?25l\033[3J\033[H',end='')
            print('')
            print('Music classifier helper')
            print('')
            print('Loading')
            print(path)
            print('All keys matched successfully')
            print('Data loader ready')
            print('Evaluating' + '.' * point2 + ' ' * (num_points - point2))
            audio = data
            outputs = model(audio)
            results.extend(torch.sigmoid(outputs).tolist())
            point2 += 1
            if point2 > num_points:
                point2 = 0

    group = frame.groupby('id')['genres'].apply(list)
    start = 0
    end = 0
    direc = 0
    memory = {}

    for line in group:
        end = len(line) + start
        memory.setdefault(direc,[]).append(numpy.mean(results[start:end], axis=0).tolist())
        start = end
        direc += 1 

    answer = []

    for case in memory.values():
       maximun = max(case[0])
       answer.append(case[0].index(maximun))

    memory.clear()
    frame = frame.iloc[0:0]
    column1.clear
    column2.clear

    point4 = 0

    for n, ans in enumerate(answer):
        genres = {  0: 'Classical',
                    1: 'Country',
                    2: 'Electronic', 
                    3: 'Metal', 
                    4: 'Jazz',
                    5: 'Pop', 
                    6: 'Hiphop',  
                    7: 'Rock' }
        if n == 0:
            print('Results: ')
        print(paths[n][limit_name:],'->',genres[ans])

    answer.clear()

    







