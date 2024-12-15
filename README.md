# Получение зависимости _MSD<sub>E</sub>(t)_ с помощью модуля _msd_E.py_
1) **msd_E.py** применяется для получения зависимости _MSD<sub>E</sub>(t)_, которая используется для оценки ионной проводимости (_σ<sub>E</sub>_) по соотношению Эйнштейна (выражения (1), (2)) (подробнее в исследовании: _ссылка будет доступна позднее_).


![image](https://github.com/user-attachments/assets/e96d6a62-5089-4ec5-8533-cd5a1b8819fc)

где _σ<sub>E</sub>_ – ионная проводимость, полученная по выражению Эйнштейна [1/(Ом∙м)], _e_ – элементарный электронный заряд [Кл], _V_ – объём системы [м<sup>3</sup>], _k<sub>b</sub>_ – постоянная Больцмана [Дж/К], _T_ – термодинамическая температура [К], _MSD<sub>E</sub>_ – аналог среднеквадратичного смещения [м<sup>2</sup>] для набора _N_ частиц в момент времени _t_ [с] на молекулярно-динамической (МД) траектории, _N_ – количество ионов в системе, _z<sub>i</sub>_ (_z<sub>j</sub>_) – формальный заряд _i_-того (_j_-того) иона, _r<sub>i</sub>(t)_ (_r<sub>j</sub>(t)_) – вектор положения _i_-того (_j_-того) иона [м] в момент времени _t_ [с] (в случае иона лития, сульфонатной группы использовался вектор положения иона, атома серы соответственно), _‹...›<sub>t0</sub>_ – усредненные значения по всей МД-траектории.
