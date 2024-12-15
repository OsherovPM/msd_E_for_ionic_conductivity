

# Модуль для расчёта зависимости MSD_E(angstrom^2) от t(ps)

# Имеется 2 типа частиц (катионы Li+ и анионы SO3- (атомы серы)).
# Для каждого типа частиц имеется файл, содержащий массивы смежного смещения (displacements) .npz:
# например,   sys_pc_313_Li_displacements.npz   для катионов лития
# и   sys_pc_313_S_displacements.npz   для анионов SO3- (атомы серы).

# импорт библиотек python
import numpy as np
import pathlib
import itertools
import collections


# вводная информация для пользователя
print('Модуль для расчёта зависимости MSD_E(angstrom^2) от t(ps)')
print('Данная версия модуля прменима для расчёта зависимости MSD_E(t) между двумя типами ионов')
print('---------------------------------')
print('---------------------------------')
print('Для правильной работы исполняемого модуля требуется отдельная папка,')
print('в которой находятся ещё две папки:')
print('  * в одной из этих папок содержится исполняемый модуль (msd_E.py),')
print('    в эту же папку будут сохраняться результаты расчёта (user_output.log),')
print('    модуль (msd_E.py) нужно запускать только из этой папки;')
print('')
print('  * в другой папке должны находится исходные данные:')
print('      файл с массивами смежного смещения для катионов (например, sys_pc_313_Li_displacements.npz)')
print('      файл с массивами смежного смещения для анионов (например, sys_pc_313_S_displacements.npz)')
print('---------------------------------')
print('---------------------------------')


# получаю начальные данные от пользователя

# получаю имя папки с исходными файлами .npz
folder_name_user_input = input('Введите имя папки c файлами, содержащими массивы смежного смещения частиц:   ')
print('')

# получаю имена файлов с массивами смежного смещения для катионов и анионов от пользователя
print('Введите полное имя файла c массивами смежного смещения для катионов,')
file_name_Li_user_input = input('например,  sys_pc_313_Li_displacements.npz  :  ')
print('')
print('Введите полное имя файла c массивами смежного смещения для анионов,')
file_name_S_user_input = input('например,  sys_pc_313_S_displacements.npz  :    ')
print('')

# получаю имя выходного файла
print('Введите полное имя выходного файла с зависимостью MSD_E(t),')
file_name_user_output = input('например,  user_output.log  :    ')
print('---------------------------------')
print('---------------------------------')

# получаю пути к файлам с массивами смежного смещения
current_path = pathlib.Path.cwd()
current_path_parent = current_path.parent

file_to_load_Li = pathlib.Path(current_path_parent, folder_name_user_input, file_name_Li_user_input, )

file_to_load_S = pathlib.Path(current_path_parent, folder_name_user_input, file_name_S_user_input, )

print('Файлы с массивами смежного смещения для двух типов ионов будут загружены из:')
print('катионы --->', file_to_load_Li)
print('')
print('анионы --->', file_to_load_S)
print('---------------------------------')
print('---------------------------------')

# получаю временной интервал между кадрами траектории молекулярной динамики
print('Временной интервал между кадрами траектории молекулярной динамики (МД)')
print('задавался пользователем перед моделированием системы,')
print('например, он может быть равен "124.99800" (пс)')
print('')
print('После ввода временного интервала начнётся загрузка файлов и расчёт')
print('(значения i, j, k, которые будут выводиться на экран, являются простым отображением хода процесса)')
print('---------------------------------')
print('---------------------------------')

time_interval_user_input = float(input('Введите временной интервал между смежными кадрами МД траектории:   '))


# загружаю исходный файл для катионов
file_loaded_Li = np.load(file_to_load_Li)
file_keys_Li = file_loaded_Li.files

loaded_displacements_Li = []
for key in file_keys_Li:
    loaded_displacements_Li.append(file_loaded_Li[key])

# загружаю исходный файл для анионов
file_loaded_S = np.load(file_to_load_S)
file_keys_S = file_loaded_S.files

loaded_displacements_S = []
for key in file_keys_S:
    loaded_displacements_S.append(file_loaded_S[key])

print('Файлы с массивами смежного смещения для двух типов ионов загружены.')


# получаю форму массивов смежного смещения:
# number_of_atoms_Li -- это число строк в нулевом массиве списка дисплесмент, то есть число атомов
number_of_atoms_Li = loaded_displacements_Li[0].shape[0]
number_of_atoms_S = loaded_displacements_S[0].shape[0]
print('Число катионов составляет:   ', number_of_atoms_Li)
print('Число анионов составляет:    ', number_of_atoms_S)

number_of_columns_Li = loaded_displacements_Li[0].shape[1]
number_of_columns_S = loaded_displacements_S[0].shape[1]
print('Число колонок в массивах смежного смещения для катионов:     ', number_of_columns_Li)
print('Число колонок в массивах смежного смещения для анионов:      ', number_of_columns_S)


if number_of_atoms_Li == number_of_atoms_S:
    print('число катионов = числу анионов')
else:
    print('предупреждение, число катионов отличается от числа анионов')


# создаю два вектора единиц и домножаю их на парциальные зарядовые числа [безразмерные] частиц
lithium_partial_charge = 1
sulfur_partial_charge = -1

charge_array_Li = np.ones(shape=number_of_atoms_Li, dtype=np.float32) * lithium_partial_charge
charge_array_S = np.ones(shape=number_of_atoms_S, dtype=np.float32) * sulfur_partial_charge


# добавляю колонку с зарядовыми числами для массивов катионов
displacements_Li = []
for displacement_matrix in loaded_displacements_Li:
    displacement_matrix_with_charges = np.c_[displacement_matrix, charge_array_Li]
    displacements_Li.append(displacement_matrix_with_charges)

# добавляю колонку с зарядовыми числами для массивов анионов
displacements_S = []
for displacement_matrix in loaded_displacements_S:
    displacement_matrix_with_charges = np.c_[displacement_matrix, charge_array_S]
    displacements_S.append(displacement_matrix_with_charges)

print('Созданы списки массивов смежного смещения с зарядовыми числами частиц')


# попарно объединияю списки с массивами смежного смещения
test_displacements_Li_S_zip = zip(displacements_Li, displacements_S)
test_displacements_Li_S_zip_list = list(test_displacements_Li_S_zip)
print('Списки с массивами смежного смещения объединены')


# получаю форму будущих массивов декартового произведения между массивами смежного смещения
cartesian_rows = (number_of_atoms_Li + number_of_atoms_S) ** 2
cartesian_columns = ((number_of_columns_Li + 1) + (number_of_columns_S + 1))

# создаю хранилище для массивов декартового произведения и заполняю его
i = 0
cartesian_product_array_adjacent_displacements_Li_S_volume = []
for matrix_Li, matrix_S in test_displacements_Li_S_zip_list:
    i += 1
    matrix_Li_above_matrix_S_array = np.r_['0,2', matrix_Li, matrix_S]
    print('i = ', i)
    matrix_Li_above_matrix_S_list = matrix_Li_above_matrix_S_array.tolist()
    cartesian_product = itertools.product(matrix_Li_above_matrix_S_list, matrix_Li_above_matrix_S_list)
    cartesian_product_list = list(cartesian_product)
    cartesian_product_array = np.array(cartesian_product_list, dtype=np.float32)
    cartesian_product_array_reshaped = cartesian_product_array.reshape((cartesian_rows, cartesian_columns))  # (n, m),
    # где n = (число_частиц_(строк)_в_массиве_Li + число_частиц_(строк)_в_массиве_S)^2,
    # а m = число_стобцов_в_матрице_Li_с_зарядом + число_стобцов_в_матрице_S_с_зарядом
    cartesian_product_array_adjacent_displacements_Li_S_volume.append(cartesian_product_array_reshaped)

print('Хранилище для массивов декартового произведения заполнено')


# получаю количество массивов декартового произведения
cartesian_product_volume_length = len(cartesian_product_array_adjacent_displacements_Li_S_volume)

# создаю своеобразный счётчик в виде словаря
# для подсчёта и последующего усреднения смещений по временным интервалам:
# ключ -- длина интервала между кадрами
# значение -- количество таких интервалов при расчёте (нужно для усреднения по этому количеству)
dict_counter_of_displacement_for_equal_time_intervals = collections.defaultdict(int)

# создаю хранилище для Эйнштейновых сумм (суммы в выражении Эйнштейна)
volume_Einstein_sum = collections.defaultdict(float)

# начинаю цикл по количеству массивов декартового произведения
j = 0
k = 0
for start_id in range(cartesian_product_volume_length):
    j += 1
    print('j = ', j)
    # создаю накопительный массив такой же формы, как и массив декартового произведения
    accumulate_displacement = np.zeros(shape=(cartesian_rows, cartesian_columns), )  # (n, m)
    # начинаю цикл от текущего номера массива декартового произведения
    # до количества массивов декартового произведения
    for itr_id in range(start_id, cartesian_product_volume_length):
        k += 1
        print('k = ', k)
        # получаю index (длину интервала между кадрами)
        index = itr_id - start_id
        # создаю в словаре ключ index и прибавляю к нему 1
        dict_counter_of_displacement_for_equal_time_intervals[index] += 1
        # если компоненты (x, y, z) в accumulate_displacement являются пустыми (заполнены только нулями),
        # то поэлементно прибавляю значения текущего массива декартового произведения к накопительному массиву,
        # иначе поэлементно добавляю значения только компонент (x, y, z) из текущего массива декартового произведения
        # к уже имеющимся значениям в накопительном массиве
        if (accumulate_displacement[:, :3].sum() + accumulate_displacement[:, 4:7].sum()) == 0:
            accumulate_displacement += cartesian_product_array_adjacent_displacements_Li_S_volume[itr_id]
        else:
            accumulate_displacement[:, :3] += cartesian_product_array_adjacent_displacements_Li_S_volume[itr_id][:, :3]
            accumulate_displacement[:, 4:7] += cartesian_product_array_adjacent_displacements_Li_S_volume[itr_id][:, 4:7]
        # начинаю проводить вычисления:
        # поэлементно умножаю первые 4 столбца (x_Li, y_Li, z_Li, заряд_Li)
        # на последние 4 столбца (x_S, y_S, z_S, заряд_S),
        # таким образом, в колонках получаются соответствующие произведения
        # компонент (x, y, z) и зарядов
        calculate_displacement = np.multiply(accumulate_displacement[:, :4], accumulate_displacement[:, 4:])
        # поэлементно складываю значения нулевой, первой и второй колонки calculate_displacement
        # и поэлементно присваиваю результат в значения нулевой колонки calculate_displacement,
        # получая, таким образом, скалярное произведение векторов смещения в выражении Эйнштейна
        calculate_displacement[:, 0] = calculate_displacement[:, 0] + calculate_displacement[:, 1] + calculate_displacement[:, 2]
        # удаляю лишние колонки (первую и вторую) в массиве calculate_displacement,
        calculate_displacement = np.delete(arr=calculate_displacement, obj=[1, 2], axis=1)
        # поэлементно умножаю скалярное произведение векторов смещения на произведение зарядов
        calculate_displacement = np.prod(calculate_displacement, axis=1)
        # складываю полученные значения в одно число, получая Эйнштейнову сумму для одного
        # конкретного интервала (между кадрами МД траектории 2_и_1 или 3_и_1, или 3_и_2 и др.)
        calculate_displacement = np.sum(calculate_displacement)
        # добавляю (прибавляю к уже имеющимся суммам, если они есть) эту сумму
        # по ключу (длине интервала между кадрами) в словарь-хранилище этих сумм
        # (например, между кадрами 2_и_1 длина интервала = 1, index = 0
        #            между кадрами 3_и_1 длина интервала = 2, index = 1
        #            между кадрами 3_и_2 длина интервала = 1, index = 0)
        volume_Einstein_sum[index] += calculate_displacement

print('Количество Эйнштейновых сумм для каждой длины интервала между кадрами:   ', dict_counter_of_displacement_for_equal_time_intervals)
print('Получены Эйнштейновы суммы для различных длин интервалов между кадрами:  ', volume_Einstein_sum)


# создаю массив для зависимости MSD_E(t)
# (число строк = количеству массивов декартового произведения в списке
# cartesian_product_array_adjacent_displacements_Li_S_volume + 1;
# число столбцов = 2 (для времени и для "MSD_E" -- член в угловых скобках в выражении Эйнштейна))
time_msd_array = np.zeros(shape=(cartesian_product_volume_length + 1, 2), )

# получаю диапазон точек на оси времени для зависимости MSD_E(t) (первая точка -- 0)
time_indexes = range(time_msd_array.shape[0])

# получаю диапазон ключей для списочного выражения ниже
dict_counter_keys = time_indexes[:-1]

# получаю список, состоящий из значений словаря dict_counter_of_displacement_for_equal_time_intervals в том же порядке,
# то есть получаю список, состоящий из количеств Эйнштейновых сумм для одинаковых интервалов между кадрами
number_of_Einstein_sums = [dict_counter_of_displacement_for_equal_time_intervals[key] for key in dict_counter_keys]

# заполняю нулевую колонку (колонка со временем) time_msd_array
# (поэлементно присваиваю значения из диапазона: 0, 1, 2, 3, ...)
time_msd_array[:, 0] = time_indexes

# поэлементно умножаю значения нулевой колонки time_msd_array на интервал времени между двумя кадрами траектории
time_msd_array[:, 0] *= time_interval_user_input

# заполняю первую колонку time_msd_array значениями из хранилища сумм Эйнштейна,
# начиная с первой строки (нулевая строка -- 0)
time_msd_array[1:, 1] = list(volume_Einstein_sum.values())

# поэлементно делю значения первой колонки
# на число Эйнштейновых сумм для каждого интервала между кадрами,
# начиная с первой строки
time_msd_array[1:, 1] /= number_of_Einstein_sums

print('Получен массив с зависимостью MSD_E(t)', time_msd_array)


# сохраняю массив time_msd_array для дальнейшего использования

# получаю заголовок для сохраняемого массива
header = 'Tau(ps)_opm,3D_MSD_Einstein(angstrom^2)_opm'

# сохраняю файл на диск,
# файл сохраняется в папку с исполняемым модулем
np.savetxt(file_name_user_output, time_msd_array, header=header, delimiter=',')

print('Массив с зависимостью MSD_E(t) сохранён')
