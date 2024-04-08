import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from itertools import islice


version = '0.9.9'

class Point:
    def __init__(self, X=None, Y=None, Z=None):
        self.X = X
        self.Y = Y
        self.Z = Z

    def traveled_distance(self):
        if len(self.X) > 1:
            return np.sqrt((np.subtract(self.X[1:], np.roll(self.X, 1)[1:])) ** 2 + (
                np.subtract(self.Y[1:], np.roll(self.Y, 1)[1:])) ** 2 + (
                               np.subtract(self.Z[1:], np.roll(self.Z, 1)[1:])) ** 2)
        else:
            return 0

    def distance(self, point):
        return np.sqrt((np.subtract(self.X, point.X)) ** 2 + (np.subtract(self.Y, point.Y)) ** 2 + (
            np.subtract(self.Z, point.Z)) ** 2)

    def dot(self, point):
        return np.multiply(self.X, point.X) + np.multiply(self.Y, point.Y) + np.multiply(self.Z, point.Z)

    def project_axis(self, axis=None):
        if axis is not None:
            if axis == 'X':
                self.Y = 0
                self.Z = 0
            elif axis == 'Y':
                self.X = 0
                self.Z = 0
            elif axis == 'Z':
                self.X = 0
                self.Y = 0
            else:
                raise ValueError('{} is not valid axis'.format(axis))

    def project_plane(self, plane=None):
        if plane is not None:
            if plane == 'XY':
                self.Z = 0
            elif plane == 'XZ':
                self.Y = 0
            elif plane == 'YZ':
                self.X = 0
            else:
                raise ValueError('{} is not valid plane'.format(plane))


class Line:
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2
        # vector coordinates
        self.X = np.subtract(self.p2.X, self.p1.X)
        self.Y = np.subtract(self.p2.Y, self.p1.Y)
        self.Z = np.subtract(self.p2.Z, self.p1.Z)
        # point instance
        self.p = Point(self.X, self.Y, self.Z)

    def length(self):
        return self.p1.distance(self.p2)

    def angle(self, line, deg=True):
        a = np.arccos((self.p.dot(line.p)) / self.length() / line.length())
        if not deg:
            return a
        else:
            return np.degrees(a)


class Model:
    def __init__(self):
        pass

    def tr_dist_int(self, point, event1, event2):
        dis = 0
        frame1 = self.frame(event1)
        frame2 = self.frame(event2)
        x = self.__data__[point + ' X'].loc[frame1]
        y = self.__data__[point + ' Y'].loc[frame1]
        z = self.__data__[point + ' Z'].loc[frame1]
        for i in range(frame1 + 1, frame2):
            x_l = x
            y_l = y
            z_l = z
            frame = i
            x = self.__data__[point + ' X'].loc[frame]
            y = self.__data__[point + ' Y'].loc[frame]
            z = self.__data__[point + ' Z'].loc[frame]
            dis = dis + ((x - x_l) ** 2 + (y - y_l) ** 2 + (z - z_l) ** 2) ** .5
        return dis

    def load_model(self, path, key_moments=['fs', 'ms', 'to']):
        self.__path__ = path
        with open(path) as fp:

            line = fp.readline()
            cnt = 1
            while 'MARKER_NAMES' not in line:
                line = fp.readline()
                cnt += 1
        fp.close()
        self.__data__ = pd.read_csv(path, sep='\t', skiprows=cnt)

        list_name_markers = ["Frame", "Time"]

        for i in self.__data__:
            if i == "Frame" or i == "Time":
                continue
            a = i[-1]
            b = i[:-1].strip()
            c = b + " " + a
            list_name_markers.append(c)


        self.__data__.columns = list_name_markers

        self.__events__ = pd.read_csv(path, sep='\t', skiprows=9, nrows=cnt - 10,
                                      names=['type', 'name', 'frame', 'time'])
        self.load_info()
        self.__data__['000 X'] = 0
        self.__data__['000 Y'] = 0
        self.__data__['000 Z'] = 0

        self.__data__['001 X'] = 1
        self.__data__['001 Y'] = 0
        self.__data__['001 Z'] = 0

        self.__data__['010 X'] = 0
        self.__data__['010 Y'] = 1
        self.__data__['010 Z'] = 0

        self.__data__['100 X'] = 0
        self.__data__['100 Y'] = 0
        self.__data__['100 Z'] = 1

        event_list = self.__events__.name.to_list()

        new_event_list = []
        for i in event_list:
            i = i[::-1]
            count_numbers = 0
            for j in i:
                if j in "0123456789":
                    count_numbers += 1
            new_event_list.append(i[count_numbers:][::-1])


        event_list = new_event_list

        fs = key_moments[0]
        ms = key_moments[1]
        to = key_moments[2]
        Lsteps = []
        Rsteps = []
        for i in event_list:
            if i == fs + '_L' or i == ms + '_L' or i == to + '_L':
                Lsteps.append(i)
            if i == fs + '_R' or i == ms + '_R' or i == to + '_R':
                Rsteps.append(i)

        try:
            while Lsteps[0] != fs + "_L":
                Lsteps = Lsteps[1:]
        except:
            pass
        try:
            while Rsteps[0] != fs + "_R":
                Rsteps = Rsteps[1:]
        except:
            pass
        cout_steps = min(len(Lsteps) // 3, len(Rsteps) // 3) - 1

        #print('Корректных шагов', max(cout_steps, 0))

        self.num_steps = max(cout_steps, 0)
        
    def st_velocity(self, point, frame, axis):

        if axis == "x":
            if frame == 0:
                vel_sq = self._get_point(point, frame + 1).X - self._get_point(point, frame).X
                vel = vel_sq
            elif frame == len(self.__data__) - 1:
                vel_sq = self._get_point(point, frame).X - self._get_point(point, frame - 1).X
                vel = vel_sq
            else:
                vel_sq = self._get_point(point, frame + 1).X - self._get_point(point, frame - 1).X
                vel = vel_sq/2
            return vel*self.__info__['FREQUENCY']

        elif axis == "y":
            if frame == 0:
                vel_sq = self._get_point(point, frame + 1).Y - self._get_point(point, frame).Y
                vel = vel_sq
            elif frame == len(self.__data__) - 1:
                vel_sq = self._get_point(point, frame).Y - self._get_point(point, frame - 1).Y
                vel = vel_sq
            else:
                vel_sq = self._get_point(point, frame + 1).Y - self._get_point(point, frame - 1).Y
                vel = vel_sq/2
            return vel*self.__info__['FREQUENCY']

        elif axis == "z":
            if frame == 0:
                vel_sq = self._get_point(point, frame + 1).Z - self._get_point(point, frame).Z
                vel = vel_sq
            elif frame == len(self.__data__) - 1:
                vel_sq = self._get_point(point, frame).Z - self._get_point(point, frame - 1).Z
                vel = vel_sq
            else:
                vel_sq = self._get_point(point, frame + 1).Z - self._get_point(point, frame - 1).Z
                vel = vel_sq/2
            return vel*self.__info__['FREQUENCY']

        else:
            if frame == 0:
                vel_sq = (self._get_point(point, frame + 1).X - self._get_point(point, frame).X)**2 + (self._get_point(point, frame + 1).Y - self._get_point(point, frame).Y)**2 + (self._get_point(point, frame + 1).Z - self._get_point(point, frame).Z)**2
                vel = (vel_sq)**.5
            elif frame == len(self.__data__) - 1:
                vel_sq = (self._get_point(point, frame).X - self._get_point(point, frame - 1).X)**2 + (self._get_point(point, frame).Y - self._get_point(point, frame - 1).Y)**2 + (self._get_point(point, frame).Z - self._get_point(point, frame - 1).Z)**2
                vel = (vel_sq)**.5
            else:
                vel_sq = (self._get_point(point, frame + 1).X - self._get_point(point, frame - 1).X)**2 + (self._get_point(point, frame + 1).Y - self._get_point(point, frame - 1).Y)**2 + (self._get_point(point, frame + 1).Z - self._get_point(point, frame - 1).Z)**2
                vel = (vel_sq/4)**.5
            return vel*self.__info__['FREQUENCY']
    
    def st_vel_int(self, point, frame1, frame2, axis=None):
        v = []
        for i in range(frame1, frame2):
            v += [self.st_velocity(point, i, axis)]
        return v

    def angl_vel_int(self, angl_list, frame1, frame2):
        v = []
        for i in range(frame2-frame1-1):
            if i == 0:
                v.append((angl_list[i + 1] - angl_list[i]) * self.__info__['FREQUENCY'])
            elif i == frame2 - 1:
                v.append((angl_list[i] - angl_list[i - 1]) * self.__info__['FREQUENCY'])
            else:
                v.append((angl_list[i + 1] - angl_list[i - 1]) / 2 * self.__info__['FREQUENCY'])
        return v

    def load_info(self):
        with open(path) as myfile:
            test_info = list(islice(myfile, 9))
        test_info = [element.replace('\n', '') for element in test_info][1:]
        myfile.close()
        self.__info__ = {}
        for el in test_info:
            p = el.split('\t')
            try:
                int(p[1])
            except:
                self.__info__.update({p[0]: p[1]})
            else:
                self.__info__.update({p[0]: int(p[1])})

    def traveled_distance(self, point):
        p1 = Point(self.__data__[point + ' X'], self.__data__[point + ' Y'], self.__data__[point + ' Z'])
        return p1.traveled_distance()

    def time(self, event):
        return self.__events__[self.__events__['name'] == event]['time'].iloc[0]

    def frame(self, event):
        return self.__events__[self.__events__['name'] == event]['frame'].iloc[0]

    def distance(self, point1, point2, frame=None, time=None, event=None):
        if all(x is None for x in [frame, time, event]):
            p1 = self._get_point(point1)
            p2 = self._get_point(point2)
            return p1.distance(p2)

        else:
            if frame:
                t = frame - 1
            elif time:
                t = int(time * self.__info__['FREQUENCY']) - 1
            else:
                t = self.__events__[self.__events__['name'] == event]['frame'].iloc[0] - 1

            p1 = self._get_point(point1, t)
            p2 = self._get_point(point2, t)
            return p1.distance(p2)

    def angle_int(self, point1, point2, point3, point4, event1, event2, plane=None):
        frame1 = self.frame(event1)
        frame2 = self.frame(event2)
        ang = []
        for i in range(frame1, frame2):
            ang.append(self.angle(point1, point2, point3, point4, frame=i, plane=plane))
        return ang

    def distance_int(self, point1, point2, event1, event2):
        frame1 = self.frame(event1)
        frame2 = self.frame(event2)
        dis = []
        for i in range(frame1, frame2):
            dis.append(self.distance(point1, point2, frame=i))
        return dis

    def angle(self, point1, point2, point3, point4=None, frame=None, time=None, event=None, deg=True, plane=None):
        if all(x is None for x in [frame, time, event]):
            if not point4:
                p1 = self._get_point(point1)
                p2 = self._get_point(point2)
                p3 = self._get_point(point3)
                p1.project_plane(plane)
                p2.project_plane(plane)
                p3.project_plane(plane)
                return self.three_point_angle(p1, p2, p3, deg)
            else:
                p1 = self._get_point(point1)
                p2 = self._get_point(point2)
                p3 = self._get_point(point3)
                p4 = self._get_point(point4)
                p1.project_plane(plane)
                p2.project_plane(plane)
                p3.project_plane(plane)
                p4.project_plane(plane)
                return self.four_point_angle(p1, p2, p3, p4, deg)
        else:
            if frame:
                t = frame - 1
            elif time:
                t = int(time * self.__info__['FREQUENCY']) - 1
            else:
                t = self.__events__[self.__events__['name'] == event]['frame'].iloc[0] - 1
            if not point4:
                p1 = self._get_point(point1, t)
                p2 = self._get_point(point2, t)
                p3 = self._get_point(point3, t)
                p1.project_plane(plane)
                p2.project_plane(plane)
                p3.project_plane(plane)
                return self.three_point_angle(p1, p2, p3, deg)
            else:
                p1 = self._get_point(point1, t)
                p2 = self._get_point(point2, t)
                p3 = self._get_point(point3, t)
                p4 = self._get_point(point4, t)
                p1.project_plane(plane)
                p2.project_plane(plane)
                p3.project_plane(plane)
                p4.project_plane(plane)
                return self.four_point_angle(p1, p2, p3, p4, deg)

    def _get_point(self, point, t=None):
        if t is not None:
            return Point(self.__data__[point + ' X'].iloc[t], self.__data__[point + ' Y'].iloc[t],
                         self.__data__[point + ' Z'].iloc[t])
        else:
            return Point(self.__data__[point + ' X'], self.__data__[point + ' Y'], self.__data__[point + ' Z'])

    def three_point_angle(self, p1, p2, p3, deg):
        l1 = Line(p2, p1)
        l2 = Line(p2, p3)
        return l1.angle(l2, deg=deg)

    def four_point_angle(self, p1, p2, p3, p4, deg):
        l1 = Line(p1, p2)
        l2 = Line(p3, p4)
        return l1.angle(l2, deg=deg)
    
    def unnumerate_events(self):
        self.__events__['name'] = self.__events__['name'].apply(lambda x: x if not x[-1].isdigit() else x[:-1] if not x[-2].isdigit() else x[:-2])
    
    def numerate_events(self):
        #print(self.__events__)
        for i, name in enumerate(self.__events__['name'].unique()):
            for j, idx in enumerate(self.__events__['name'][self.__events__['name'] == name].index):
                self.__events__['name'].at[idx] = self.__events__['name'].at[idx] + str(j + 1)

    def strip_data(self, before=None, behind=None):
        if behind is None:
            behind = int(self.__info__['FREQUENCY'] / 2)
        if before is None:
            before = int(self.__info__['FREQUENCY'] / 2)
        before_idx = self.__events__['frame'].iloc[0] - before if self.__events__['frame'].iloc[0] - before > 0 else 0
        behind_idx = self.__events__['frame'].iloc[-1] + behind if self.__events__['frame'].iloc[0] + behind < len(
            self.__data__) - 1 else len(self.__data__) - 1
        self.__data__ = self.__data__[before_idx:behind_idx]

    def angle_int_plane(self, point1, point2, point3, point4, event1, event2, plane):
        frame1 = self.frame(event1)
        frame2 = self.frame(event2)
        ang = []
        for i in range(frame1, frame2):
            p1 = self._get_point(point1, i)
            p2 = self._get_point(point2, i)
            p3 = self._get_point(point3, i)
            p4 = self._get_point(point4, i)
            x1, y1, z1 = p1.X, p1.Y, p1.Z
            x2, y2, z2 = p2.X, p2.Y, p2.Z
            x3, y3, z3 = p3.X, p3.Y, p3.Z
            x4, y4, z4 = p4.X, p4.Y, p4.Z
            if plane == 'XZ':
                ang1 = math.atan((x2 - x1) / (z1 - z2))
                ang2 = math.atan((x4 - x3) / (z3 - z4))
                ang += [np.degrees(ang1) - np.degrees(ang2)]
            if plane == 'YZ':
                ang1 = math.atan((y2 - y1) / (z2 - z1))
                ang2 = math.atan((y4 - y3) / (z4 - z3))
                ang += [np.degrees(ang1) - np.degrees(ang2)]
        return ang


def main():
    global speed, height, steps_from, steps_to, stat_path, model_path
    
    if not check_stat.get():
        path = path_stat_tf.get()
        stat_path = path.split('/')[-1]
        stat = Model()
        stat.load_model(path)
        stat_events = stat.__events__.name.to_list()
        if stat_events[-1][-1] not in '0123456789':
            stat.numerate_events()
        stat.strip_data()
    else:
        stat_path = 'Не использован'
    path = path_din_tf.get()
    model_path = path.split('/')[-1]
    m = Model()
    m.load_model(path, key_moments)
    #print(m.__events__)
    m.__events__ = m.__events__[m.__events__['name'].str.contains('|'.join(['begin',fs, ms, to]))]
    #print(m.__events__)
    m_events = m.__events__.name.to_list()
    if unnumerate_var.get() == "Да":
        m.unnumerate_events()
        #print(m.__events__)
        #print(0)
        m.numerate_events()
        #print(m.__events__)
    elif m_events[-1][-1] not in '0123456789':
        m.numerate_events()
    #print(1)
    m.strip_data()
    print(m.__info__['FREQUENCY'])

    if speed_flag.get() == km:
        speed = float(speed_tf.get())
    elif speed_flag.get() == miles:
        speed = float(speed_tf.get()) * 0.621371
    else:
        speed = float(speed_tf.get()) / 3.6

    height = float(height_tf.get())

    steps_from = int(steps_from_cb.get())
    steps_to = int(steps_to_cb.get())

    #print(m.__data__)
    #print(Back_T + ' X')
    #print(m.__data__[Back_T + ' X'])


    # Предобработка статики --------------------------------------------------------------------------------

    #print(stat.__data__)
    
    left_gol_st = 0
    right_gol_st = 0
    left_elbow_st = 180
    right_elbow_st = 180
    korp_vert_vpered_st = 0
    korp_vert_left_st = 0
    korp_bed_left_st = 0
    korp_bed_right_st = 0
    bed_vert_left_st = 0
    bed_vert_right_st = 0
    gol_op_left_st = 90
    gol_op_right_st = 90
    zero_z_L = 0
    zero_z_R = 0
    stat_hip_ext_ang_L = 0
    stat_hip_ext_ang_R = 0
    stat_should_left_xz = 0
    stat_should_right_xz = 0
    stat_should_left_yz = 0
    stat_should_right_yz = 0
    stat_Hip_Knee_Ankle_L = 180
    stat_Hip_Knee_Ankle_R = 180
    
    if not check_stat.get():
        left_gol_st = stat.angle(Knee_L, Ankle_L, Toe_L, event='static_stand1')
        right_gol_st = stat.angle(Knee_R, Ankle_R, Toe_R, event='static_stand1')
        left_elbow_st = stat.angle(Shoulder_L, Elbow_L, Wrist_L, event='static_stand1')
        right_elbow_st = stat.angle(Shoulder_R, Elbow_R, Wrist_R, event='static_stand1')
        print(right_elbow_st, left_elbow_st)
        ang_z = stat.angle(Back_B, Back_T, '000', '100', event='static_stand1')
        ang_x = stat.angle(Back_B, Back_T, '000', '001', event='static_stand1')
        ang_y = stat.angle(Back_B, Back_T, '000', '010', event='static_stand1')
        gip = stat.distance(Back_B, Back_T, event='static_stand1')
        cat_z = gip * math.cos(ang_z * 3.1416 / 180)
        cat_x = gip * math.cos(ang_x * 3.1416 / 180)
        cat_y = gip * math.cos(ang_y * 3.1416 / 180)
        korp_vert_vpered_st = math.atan(cat_x / cat_z) * 180 / 3.1416
        korp_vert_left_st = math.atan(cat_y / cat_z) * 180 / 3.1416
        korp_bed_left_st = stat.angle(Knee_L, Hip_L, Back_B, Back_T, event='static_stand1')
        korp_bed_right_st = stat.angle(Knee_R, Hip_R, Back_B, Back_T, event='static_stand1')
        #print(korp_bed_left_st, korp_bed_right_st)
        ang_z = stat.angle(Knee_L, Hip_L, '000', '100', event='static_stand1')
        ang_x = stat.angle(Knee_L, Hip_L, '000', '001', event='static_stand1')
        gip = stat.distance(Knee_L, Hip_L, event='static_stand1')
        cat_z = gip * math.cos(ang_z * 3.1416 / 180)
        cat_x = gip * math.cos(ang_x * 3.1416 / 180)
        bed_vert_left_st = math.atan(cat_x / cat_z) * 180 / 3.1416
        ang_z = stat.angle(Knee_R, Hip_R, '000', '100', event='static_stand1')
        ang_x = stat.angle(Knee_R, Hip_R, '000', '001', event='static_stand1')
        gip = stat.distance(Knee_R, Hip_R, event='static_stand1')
        cat_z = gip * math.cos(ang_z * 3.1416 / 180)
        cat_x = gip * math.cos(ang_x * 3.1416 / 180)
        bed_vert_right_st = math.atan(cat_x / cat_z) * 180 / 3.1416
        gol_op_left_st = stat.angle(Ankle_L, Knee_L, '000', '001', event='static_stand1')
        gol_op_right_st = stat.angle(Ankle_R, Knee_R, '000', '001', event='static_stand1')
        zero_z_L = stat._get_point(Heel_L, stat.frame('static_stand1')).Z
        zero_z_R = stat._get_point(Heel_R, stat.frame('static_stand1')).Z
        stat_hip_ext_ang_L = stat.angle(Knee_L, Hip_L, '000', '100', event='static_stand1', plane='XZ')
        stat_hip_ext_ang_R = stat.angle(Knee_R, Hip_R, '000', '100', event='static_stand1', plane='XZ')
        stat_should_left_xz = stat.angle(Back_T, Back_B, Shoulder_L, Elbow_L, event='static_stand1', plane='XZ')
        stat_should_right_xz = stat.angle(Back_T, Back_B, Shoulder_R, Elbow_R, event='static_stand1', plane='XZ')
        stat_should_left_yz = stat.angle(Back_T, Back_B, Shoulder_L, Elbow_L, event='static_stand1', plane='YZ')
        stat_should_right_yz = stat.angle(Back_T, Back_B, Shoulder_R, Elbow_R, event='static_stand1', plane='YZ')
        stat_Hip_Knee_Ankle_L = stat.angle(Hip_L, Knee_L, Ankle_L, event='static_stand1')
        stat_Hip_Knee_Ankle_R = stat.angle(Hip_R, Knee_R, Ankle_R, event='static_stand1')

    #

    canvas = Canvas(frame_calculations, width=1400, height=650)
    canvas.pack(anchor=CENTER, expand=1)
    canvas.create_rectangle(30, 35, 325, 167)
    canvas.create_rectangle(360, 35, 800, 170)
    canvas.create_rectangle(830, 35, 1240, 220)
    canvas.create_rectangle(30, 220, 455, 315)
    canvas.create_rectangle(475, 220, 785, 325)
    canvas.create_rectangle(830, 250, 1270, 355)
    canvas.create_rectangle(30, 345, 480, 460)
    # canvas.create_rectangle(430, 366, 900, 516)

    zagalovok = ttk.Label(frame_calculations, text="Результаты обработки файлов")
    zagalovok.place(relx=0.43, rely=0.01)

    # Предобработка для фаз --------------------------------------------------------------------------------
    global step_duration_np, stance_phase_np, fly_phase_np, depreciation_phase_np, repulsion_phase_np, T_step
    global double_contact_phase_L_np, double_contact_phase_R_np, single_contact_phase_L_np, single_contact_phase_R_np

    step_duration_L = []
    step_duration_R = []

    for i in range(m.num_steps - 1):
        step_duration_L += [m.time(fs + '_L' + str(i + 2)) - m.time(fs + '_L' + str(i + 1))]
        step_duration_R += [m.time(fs + '_R' + str(i + 2)) - m.time(fs + '_R' + str(i + 1))]

    stance_phase_L = []
    stance_phase_R = []

    for i in range(1, m.num_steps + 1):
        stance_phase_L += [m.time(to + '_L' + str(i)) - m.time(fs + '_L' + str(i))]
        stance_phase_R += [m.time(to + '_R' + str(i)) - m.time(fs + '_R' + str(i))]

    fly_phase_L = []
    fly_phase_R = []
    
    double_contact_phase_L = []
    double_contact_phase_R = []
    
    single_contact_phase_L = []
    single_contact_phase_R = []
        
    #if (type_move.get() == 'Бег'):
    if (m.time(fs + '_R1') > m.time(fs + '_L1')):
        for i in range(1, m.num_steps):
            fly_phase_L += [m.time(fs + '_R' + str(i)) - m.time(to + '_L' + str(i))]
            fly_phase_R += [m.time(fs + '_L' + str(i + 1)) - m.time(to + '_R' + str(i))]
    else:
        for i in range(1, m.num_steps):
            fly_phase_L += [m.time(fs + '_R' + str(i + 1)) - m.time(to + '_L' + str(i))]
            fly_phase_R += [m.time(fs + '_L' + str(i)) - m.time(to + '_R' + str(i))]
    #elif (type_move.get() == 'Ходьба'):
    if (m.time(fs + '_R1') > m.time(fs + '_L1')):
        for i in range(1, m.num_steps):
            double_contact_phase_L += [m.time(to + '_R' + str(i)) - m.time(fs + '_L' + str(i + 1))]
            double_contact_phase_R += [m.time(to + '_L' + str(i)) - m.time(fs + '_R' + str(i))]
    else:
        for i in range(1, m.num_steps):
            double_contact_phase_L += [m.time(to + '_R' + str(i)) - m.time(fs + '_L' + str(i))]
            double_contact_phase_R += [m.time(to + '_L' + str(i)) - m.time(fs + '_R' + str(i + 1))]
    
    for i in range(1, m.num_steps):
        single_contact_phase_L += [m.time(fs + '_R' + str(i + 1)) - m.time(to + '_R' + str(i))]
        single_contact_phase_R += [m.time(fs + '_L' + str(i + 1)) - m.time(to + '_L' + str(i))]
    
    #print(double_contact_phase_L)
    #print(double_contact_phase_R)
    #print(fly_phase_L)
    #print(fly_phase_R)

    depreciation_phase_L = []
    depreciation_phase_R = []

    for i in range(1, m.num_steps + 1):
        depreciation_phase_L += [m.time(ms + '_L' + str(i)) - m.time(fs + '_L' + str(i))]
        depreciation_phase_R += [m.time(ms + '_R' + str(i)) - m.time(fs + '_R' + str(i))]

    repulsion_phase_L = []
    repulsion_phase_R = []

    for i in range(1, m.num_steps + 1):
        repulsion_phase_L += [m.time(to + '_L' + str(i)) - m.time(ms + '_L' + str(i))]
        repulsion_phase_R += [m.time(to + '_R' + str(i)) - m.time(ms + '_R' + str(i))]

    step_duration_np = np.array(step_duration_L[steps_from:steps_to] + step_duration_R[steps_from:steps_to])
    stance_phase_np = np.array(stance_phase_L[steps_from:steps_to] + stance_phase_R[steps_from:steps_to])
    fly_phase_np = np.array(fly_phase_L[steps_from:steps_to] + fly_phase_R[steps_from:steps_to])
    depreciation_phase_np = np.array(
        depreciation_phase_L[steps_from:steps_to] + depreciation_phase_R[steps_from:steps_to])
    repulsion_phase_np = np.array(repulsion_phase_L[steps_from:steps_to] + repulsion_phase_R[steps_from:steps_to])
    
    double_contact_phase_L_np = np.array(double_contact_phase_L[steps_from:steps_to])
    double_contact_phase_R_np = np.array(double_contact_phase_R[steps_from:steps_to])
    single_contact_phase_L_np = np.array(single_contact_phase_L[steps_from:steps_to])
    single_contact_phase_R_np = np.array(single_contact_phase_R[steps_from:steps_to])
    
    step_frequency = 1 / np.mean(np.array(step_duration_L[steps_from:steps_to]))
    if check_parameter_1.get():
        step_frequency_lb = ttk.Label(frame_calculations, text="Частота шагов:      " + str(step_frequency)[:4] + " гц")
    else:
        step_frequency_lb = ttk.Label(frame_calculations, text="Частота шагов:      ---")
    step_frequency_lb.place(relx=0.03, rely=0.06)


    step_length = np.mean(step_duration_np)*speed/3.6*100
    if check_parameter_2.get():
        step_length_lb = ttk.Label(frame_calculations, text="Длина шага:      " + str(step_length) + " см,   " + str(int(step_length * 100 / height)) + " %")
    else:
        step_length_lb = ttk.Label(frame_calculations, text="Длина шага:      ---")
    step_length_lb.place(relx=0.03, rely=0.09)

    T_step = np.mean(step_duration_np)/2

    stance_phase = np.mean(stance_phase_L[steps_from:steps_to] + stance_phase_R[steps_from:steps_to])
    if check_parameter_3.get():
        stance_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы опоры:      " + str(stance_phase)[:4] + " с,   " + str(int(stance_phase * 100 / T_step)) + " %")
    else:
        stance_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы опоры:      ---")
    stance_phase_lb.place(relx=0.03, rely=0.12)

    fly_phase = np.mean(fly_phase_L[steps_from:steps_to] + fly_phase_R[steps_from:steps_to])
    if check_parameter_4.get():
        fly_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы полета:      " + str(fly_phase)[:4] + " с,   " + str(int(fly_phase * 100 / T_step)) + " %")
    else:
        fly_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы полета:      ---")
    fly_phase_lb.place(relx=0.03, rely=0.15)

    depreciation_phase = np.mean(depreciation_phase_R[steps_from:steps_to] + depreciation_phase_L[steps_from:steps_to])
    if check_parameter_5.get():
        depreciation_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы амортизации:      " + str(depreciation_phase)[:4] + " с,   " + str(int(depreciation_phase * 100 / T_step)) + " %")
    else:
        depreciation_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы амортизации:      ---")
    depreciation_phase_lb.place(relx=0.03, rely=0.18)


    repulsion_phase = np.mean(repulsion_phase_R[steps_from:steps_to] + repulsion_phase_L[steps_from:steps_to])
    if check_parameter_6.get():
        repulsion_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы отталкивания:      " + str(repulsion_phase)[:4] + " с,   " + str(int(repulsion_phase * 100 / T_step)) + " %")
    else:
        repulsion_phase_lb = ttk.Label(frame_calculations, text="Длительность фазы отталкивания:      ---")
    repulsion_phase_lb.place(relx=0.03, rely=0.21)

    # Предобработка для движений корпуса --------------------------------------------------------------------------------
    global vertical_swing_np, sr_xz_np, mi_xz_np, ma_xz_np, max_R, max_L, shoulder_ampl_L_np, shoulder_ampl_R_np, hip_ampl_L_np, hip_ampl_R_np, hip_ampl_np, shoulder_ampl_np
    global swing_ampl_np

    vertical_swing = []
    for i in range(2, m.num_steps + 1):
        vertical_trj = m.distance_int(Back_B, '000', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        vertical_swing += [max(vertical_trj) - min(vertical_trj)]
    ma_xz = []
    mi_xz = []
    am_xz = []
    sr_xz = []
    ma_yz = []
    mi_yz = []
    am_yz = []
    sr_yz = []
    for i in range(2, m.num_steps + 1):
        ang_z = m.angle_int(Back_B, Back_T, '000', '100', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        ang_x = m.angle_int(Back_B, Back_T, '000', '001', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        ang_y = m.angle_int(Back_B, Back_T, '000', '010', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        gip = m.distance_int(Back_B, Back_T, fs + '_L' + str(i - 1), fs + '_L' + str(i))
        cat_z = [0] * len(gip)
        cat_x = [0] * len(gip)
        cat_y = [0] * len(gip)
        ang_xz = [0] * len(gip)
        ang_yz = [0] * len(gip)
        for i in range(len(gip)):
            cat_z[i] = gip[i] * math.cos(ang_z[i] * 3.1416 / 180)
            cat_x[i] = gip[i] * math.cos(ang_x[i] * 3.1416 / 180)
            cat_y[i] = gip[i] * math.cos(ang_y[i] * 3.1416 / 180)
            ang_xz[i] = math.atan(cat_x[i] / cat_z[i]) * 180 / 3.1416
            ang_yz[i] = math.atan(cat_y[i] / cat_z[i]) * 180 / 3.1416
        maxi = max(ang_xz)
        mini = min(ang_xz)
        ampl_xz = maxi - mini
        ma_xz.append(maxi)
        mi_xz.append(mini)
        am_xz.append(ampl_xz)
        sr_xz.append(np.mean(np.array(ang_xz)))

        maxi = max(ang_yz)
        mini = min(ang_yz)
        ampl_yz = maxi - mini
        ma_yz.append(maxi)
        mi_yz.append(mini)
        am_yz.append(ampl_yz)
        sr_yz.append(np.mean(np.array(ang_yz)))
    shoulder_ampl = []
    shoulder_ampl_L = []
    shoulder_ampl_R = []
    hip_ampl = []
    hip_ampl_L = []
    hip_ampl_R = []
    swing_ampl = []
    for i in range(2, m.num_steps + 1):
        shoulder = m.angle_int(Shoulder_L, Shoulder_R, '000', '001', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        shoulder_L = m.angle_int(Shoulder_L, Back_T, '000', '001', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        shoulder_R = m.angle_int(Back_T, Shoulder_R, '000', '001', fs + '_R' + str(i - 1), fs + '_R' + str(i))
        shoulder_ampl.append(max(shoulder) - min(shoulder))
        shoulder_ampl_L += [max(shoulder_L) - min(shoulder_L)]
        shoulder_ampl_R += [max(shoulder_R) - min(shoulder_R)]
        hips = m.angle_int(Hip_L, Hip_R, '000', '001', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        hips_L = m.angle_int(Hip_L, Back_B, '000', '001', fs + '_L' + str(i - 1), fs + '_L' + str(i))
        hips_R = m.angle_int(Back_B, Hip_R, '000', '001', fs + '_R' + str(i - 1), fs + '_R' + str(i))
        hip_ampl.append(max(hips) - min(hips))
        hip_ampl_L += [max(hips_L) - min(hips_L)]
        hip_ampl_R += [max(hips_R) - min(hips_R)]
        swing_ampl += [max(m.angle_int(Shoulder_R, Shoulder_L, Hip_R, Hip_L, fs + '_R' + str(i - 1), fs + '_R' + str(i)))]


    vertical_swing_np = np.array(vertical_swing[steps_from:steps_to])/10
    if check_parameter_7.get():
        vertical_swing_lb = ttk.Label(frame_calculations, text="Размах вертикальных колебаний:      " + format(float(str(vertical_swing_np[0])), '.1f') + " см,   " + str(int(vertical_swing_np[0] * 100 / height)) + " %")
    else:
        vertical_swing_lb = ttk.Label(frame_calculations, text="Размах вертикальных колебаний:      ---")
    vertical_swing_lb.place(relx=0.27, rely=0.06)

    if direction == 'x':
        ma_xz_np = np.array(ma_xz[steps_from:steps_to])
        mi_xz_np = np.array(mi_xz[steps_from:steps_to])
        am_xz_np = np.array(am_xz[steps_from:steps_to])
        sr_xz_np = np.array(sr_xz[steps_from:steps_to])
        max_R = -np.array(mi_yz[steps_from:steps_to])
        max_L = np.array(ma_yz[steps_from:steps_to])
    elif direction == '-x':
        ma_xz_np = -np.array(mi_xz[steps_from:steps_to]) - korp_vert_vpered_st
        mi_xz_np = -np.array(ma_xz[steps_from:steps_to]) - korp_vert_vpered_st
        am_xz_np = np.array(am_xz[steps_from:steps_to])
        sr_xz_np = -np.array(sr_xz[steps_from:steps_to]) - korp_vert_vpered_st
        max_R = np.array(ma_yz[steps_from:steps_to]) - korp_vert_left_st
        max_L = -np.array(mi_yz[steps_from:steps_to]) + korp_vert_left_st

    if check_parameter_8.get():
        angle_forw_back_lb = ttk.Label(frame_calculations, text="Угол наклона корпуса вперед-назад [град.]:\naverage:    " + format(float(str(sr_xz_np[0])), '.1f') + "    min:    " + format(float(str(mi_xz_np[0])), '.1f') + "    max:    " + format(float(str(ma_xz_np[0])), '.1f'))
    else:
        angle_forw_back_lb = ttk.Label(frame_calculations, text="Угол наклона корпуса вперед-назад:      ---")
    angle_forw_back_lb.place(relx=0.27, rely=0.09)

    if check_parameter_9.get():
        angle_right_left_lb = ttk.Label(frame_calculations, text="Угол наклона корпуса вправо-влево [град.]:\nmax_R:    " + format(float(str(max_R[0])), '.1f') + "    max_L:    " + format(float(str(max_L[0])), '.1f') + "    (max_R+max_L)/2:    " + format(float(str((max_R[0] + max_L[0]) / 2)), '.1f') + "    max_R/max_L:    " + format(float(str(max_R[0] / max_L[0])), '.1f'))
    else:
        angle_right_left_lb = ttk.Label(frame_calculations, text="Угол наклона корпуса вправо-влево:      ---")
    angle_right_left_lb.place(relx=0.27, rely=0.15)

    swing_ampl_np = np.array(swing_ampl[steps_from:steps_to])
    shoulder_ampl_np = np.array(shoulder_ampl[steps_from:steps_to])
    shoulder_ampl_L_np = np.array(shoulder_ampl_L[steps_from:steps_to])
    shoulder_ampl_R_np = np.array(shoulder_ampl_R[steps_from:steps_to])
    hip_ampl_np = np.array(hip_ampl[steps_from:steps_to])
    hip_ampl_L_np = np.array(hip_ampl_L[steps_from:steps_to])
    hip_ampl_R_np = np.array(hip_ampl_R[steps_from:steps_to])
    if check_parameter_10.get():
        rotation_amplitude_lb = ttk.Label(frame_calculations,text="Амплитуда вращения таза:      " + format(float(str(np.mean(np.array(hip_ampl[steps_from:steps_to])))), '.1f') + " град")
    else:
        rotation_amplitude_lb = ttk.Label(frame_calculations, text="Амплитуда вращения таза:      ---")
    rotation_amplitude_lb.place(relx=0.27, rely=0.21)

    if check_parameter_11.get():
        rotation_amplitude_shoulders_lb = ttk.Label(frame_calculations, text="Амплитуда вращения плечевого пояса:      " + format(float(str(np.mean(np.array(shoulder_ampl[steps_from:steps_to])))), '.1f') + " град")
    else:
        rotation_amplitude_shoulders_lb = ttk.Label(frame_calculations, text="Амплитуда вращения плечевого пояса:      ---")
    rotation_amplitude_shoulders_lb.place(relx=0.27, rely=0.24)

    # Предобработка движение рук    --------------------------------------------------------------------------------
    global min_ang_elbow_left_np, max_ang_elbow_left_np, mean_ang_elbow_left_np, min_ang_elbow_right_np, max_ang_elbow_right_np, mean_ang_elbow_right_np
    global wrist_dist_r_np, wrist_dist_l_np, should_flex_ang_max_L_np, should_flex_ang_max_R_np, should_ext_ang_max_L_np, should_ext_ang_max_R_np
    global should_abd_ang_max_L_np, should_abd_ang_max_R_np, ampl_R_np, ampl_L_np

    wrist_dist_l = []
    wrist_dist_r = []
    max_ang_elbow_left = []
    min_ang_elbow_left = []
    mean_ang_elbow_left = []
    max_ang_elbow_right = []
    min_ang_elbow_right = []
    mean_ang_elbow_right = []
    should_flex_ang_max_L = []
    should_flex_ang_max_R = []
    should_ext_ang_max_L = []
    should_ext_ang_max_R = []
    should_abd_ang_max_L = []
    should_abd_ang_max_R = []
    ampl_R = []
    ampl_L = []
    
    for i in range(2, m.num_steps + 1):
        W_R = []
        W_L = []
        for j in range(m.frame(fs + '_L' + str(i - 1)), m.frame(fs + '_L' + str(i))):
            W_R += [m.__data__[Wrist_R + ' Z'].loc[j]]
            W_L += [m.__data__[Wrist_L + ' Z'].loc[j]]
        ampl_R += [max(W_R) - min(W_R)]
        ampl_L += [max(W_L) - min(W_L)]
        
        wrist_dist_r.append(m.tr_dist_int(Wrist_R, fs + '_L' + str(i - 1), fs + '_L' + str(i)))
        wrist_dist_l.append(m.tr_dist_int(Wrist_L, fs + '_L' + str(i - 1), fs + '_L' + str(i)))
        ang_left = m.angle_int(Wrist_L, Elbow_L, Shoulder_L, None, fs + '_L' + str(i - 1), fs + '_L' + str(i))
        maxi_left = max(ang_left)
        mini_left = min(ang_left)
        max_ang_elbow_left.append(maxi_left)
        min_ang_elbow_left.append(mini_left)
        mean_ang_elbow_left.append(np.mean(np.array(ang_left)))
        ang_right = m.angle_int(Wrist_R, Elbow_R, Shoulder_R, None, fs + '_L' + str(i - 1), fs + '_L' + str(i))
        maxi_right = max(ang_right)
        mini_right = min(ang_right)
        max_ang_elbow_right.append(maxi_right)
        min_ang_elbow_right.append(mini_right)
        mean_ang_elbow_right.append(np.mean(np.array(ang_right)))
        should_ext_ang_max_L += [-min(
            m.angle_int_plane(Back_T, Back_B, Shoulder_L, Elbow_L, fs + '_L' + str(i - 1), fs + '_L' + str(i),
                              plane='XZ'))]
        should_ext_ang_max_R += [-min(
            m.angle_int_plane(Back_T, Back_B, Shoulder_R, Elbow_R, fs + '_R' + str(i - 1), fs + '_R' + str(i),
                              plane='XZ'))]
        should_flex_ang_max_L += [
            max(m.angle_int_plane(Back_T, Back_B, Shoulder_L, Elbow_L, fs + '_L' + str(i - 1), fs + '_L' + str(i),
                                  plane='XZ'))]
        should_flex_ang_max_R += [
            max(m.angle_int_plane(Back_T, Back_B, Shoulder_R, Elbow_R, fs + '_R' + str(i - 1), fs + '_R' + str(i),
                                  plane='XZ'))]
        should_abd_ang_max_L += [-min(
            m.angle_int_plane(Back_T, Back_B, Shoulder_L, Elbow_L, fs + '_L' + str(i - 1), fs + '_L' + str(i),
                              plane='YZ'))]
        should_abd_ang_max_R += [
            max(m.angle_int_plane(Back_T, Back_B, Shoulder_R, Elbow_R, fs + '_R' + str(i - 1), fs + '_R' + str(i),
                                  plane='YZ'))]

    ampl_R_np = np.array(ampl_R[steps_from:steps_to])
    ampl_L_np = np.array(ampl_L[steps_from:steps_to])
    
    min_ang_elbow_left_np = np.array(min_ang_elbow_left[steps_from:steps_to]) + (180 - left_elbow_st)
    max_ang_elbow_left_np = np.array(max_ang_elbow_left[steps_from:steps_to]) + (180 - left_elbow_st)
    mean_ang_elbow_left_np = np.array(mean_ang_elbow_left[steps_from:steps_to]) + (180 - left_elbow_st)

    min_ang_elbow_right_np = np.array(min_ang_elbow_right[steps_from:steps_to]) + (180 - right_elbow_st)
    max_ang_elbow_right_np = np.array(max_ang_elbow_right[steps_from:steps_to]) + (180 - right_elbow_st)
    mean_ang_elbow_right_np = np.array(mean_ang_elbow_right[steps_from:steps_to]) + (180 - right_elbow_st)

    wrist_dist_r_np = np.array(wrist_dist_r[steps_from:steps_to])
    wrist_dist_l_np = np.array(wrist_dist_l[steps_from:steps_to])

    should_flex_ang_max_L_np = np.array(should_flex_ang_max_L[steps_from:steps_to]) + stat_should_left_xz
    should_flex_ang_max_R_np = np.array(should_flex_ang_max_R[steps_from:steps_to]) + stat_should_right_xz
    should_ext_ang_max_L_np = np.array(should_ext_ang_max_L[steps_from:steps_to]) - stat_should_left_xz
    should_ext_ang_max_R_np = np.array(should_ext_ang_max_R[steps_from:steps_to]) - stat_should_right_xz

    should_abd_ang_max_L_np = np.array(should_abd_ang_max_L[steps_from:steps_to]) - stat_should_left_yz
    should_abd_ang_max_R_np = np.array(should_abd_ang_max_R[steps_from:steps_to]) - stat_should_right_yz

    if check_parameter_12.get():
        angle_elbow_lb = ttk.Label(frame_calculations, text="Угол в локтевом суставе [град.]: average, min, max\nLeft    average:    " + format(float(str(np.mean(mean_ang_elbow_left_np))), '.1f') + "    min:    " + format(float(str(np.mean(min_ang_elbow_left_np))), '.1f') + "    max:    " + format(float(str(np.mean(max_ang_elbow_left_np))), '.1f') + "\nRight    average:    " + format(float(str(np.mean(mean_ang_elbow_right_np))), '.1f') + "    min:    " + format(float(str(np.mean(min_ang_elbow_right_np))), '.1f') + "    max:    " + format(float(str(np.mean(max_ang_elbow_right_np))), '.1f'))
    else:
        angle_elbow_lb = ttk.Label(frame_calculations, text="Угол в локтевом суставе [град.]: average, min, max\nLeft    average:    ---    min:    ---    max:    ---\nRight    average:    ---    min:    ---    max:    ---")
    angle_elbow_lb.place(relx=0.60, rely=0.06)

    if check_parameter_13.get():
        angle_shoulder_flexion_lb = ttk.Label(frame_calculations, text="Угол сгибания плеча [град.]:\nmax_R:    " + '---' + "    max_L:    " + '---' + "    (max_R+max_L)/2:    " + '---' + "    max_R/max_L:    " + '---')
    else:
        angle_shoulder_flexion_lb = ttk.Label(frame_calculations, text="Угол сгибания плеча [град.]:\nmax_R:      ---      max_L:      ---      (max_R+max_L)/2:      ---      max_R/max_L:      ---")
    angle_shoulder_flexion_lb.place(relx=0.6, rely=0.14)

    if check_parameter_14.get():
        angle_shoulder_extension_lb = ttk.Label(frame_calculations, text="Угол разгибания плеча [град.]:\nmin_R:    " + '---' + "    min_L:    " + '---' + "    (min_R+min_L)/2:    " + '---' + "    min_R/min_L:    " + '---')
    else:
        angle_shoulder_extension_lb = ttk.Label(frame_calculations, text="Угол разгибания плеча [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_shoulder_extension_lb.place(relx=0.6, rely=0.19)

    if check_parameter_15.get():
        angle_shoulder_deflection_lb = ttk.Label(frame_calculations, text="Угол отведения плеча [град.]:\nmax_R:    " + '---' + "    max_L:    " + '---' + "    (max_R+max_L)/2:    " + '---' + "    max_R/max_L:    " + '---')
    else:
        angle_shoulder_deflection_lb = ttk.Label(frame_calculations, text="Угол отведения плеча [град.]:\nmax_R:      ---      max_L:      ---      (max_R+max_L)/2:      ---      max_R/max_L:      ---")
    angle_shoulder_deflection_lb.place(relx=0.6, rely=0.24)

    if check_parameter_16.get():
        length_elbow_lb = ttk.Label(frame_calculations, text="Длина траектории кисти [см]:\nR:      " + format(float(str(np.mean(wrist_dist_r_np)/10)), '.1f') + "      L:      " + format(float(str(np.mean(wrist_dist_l_np)/10)), '.1f') + "      (R+L)/2:      " + format(float(str(np.mean(wrist_dist_r_np)/20 + np.mean(wrist_dist_l_np)/20)), '.1f') + "      R/L:      " + format(float(str(np.mean(wrist_dist_r_np)/np.mean(wrist_dist_l_np))), '.1f'))
    else:
        length_elbow_lb = ttk.Label(frame_calculations, text="Длина траектории кисти [см]:\nR:      ---      L:      ---      (R+L)/2:      ---      R/L:      ---")
    length_elbow_lb.place(relx=0.6, rely=0.29)



    # Предобработка движение маховой ноги --------------------------------------------------------------------------------
    global min_ang_ankle_r_np, min_ang_ankle_l_np, max_hip_ext_ang_L_np, max_hip_ext_ang_R_np

    min_ang_ankle_r = []
    min_ang_ankle_l = []
    max_hip_ext_ang_L = []
    max_hip_ext_ang_R = []
    for i in range(2, m.num_steps + 1):
        min_ang_ankle_l += [min(m.angle_int(Hip_L, Knee_L, Ankle_L, None, fs + '_L' + str(i - 1), fs + '_L' + str(i)))]
        min_ang_ankle_r += [min(m.angle_int(Hip_R, Knee_R, Ankle_R, None, fs + '_R' + str(i - 1), fs + '_R' + str(i)))]
        max_hip_ext_ang_L += [
            max(m.angle_int(Knee_L, Hip_L, '000', '100', fs + '_L' + str(i - 1), fs + '_L' + str(i), plane='XZ'))]
        max_hip_ext_ang_R += [
            max(m.angle_int(Knee_R, Hip_R, '000', '100', fs + '_R' + str(i - 1), fs + '_R' + str(i), plane='XZ'))]

    min_ang_ankle_r_np = np.array(min_ang_ankle_r[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_R)
    min_ang_ankle_l_np = np.array(min_ang_ankle_l[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_L)
    max_hip_ext_ang_L_np = np.array(max_hip_ext_ang_L[steps_from:steps_to]) - stat_hip_ext_ang_L
    max_hip_ext_ang_R_np = np.array(max_hip_ext_ang_R[steps_from:steps_to]) - stat_hip_ext_ang_R

    if check_parameter_17.get():
        angle_folding_lb = ttk.Label(frame_calculations, text="Угол складывания голени [град.]:\nmin_R:    " + format(float(str(np.mean(min_ang_ankle_r_np))), '.1f') + "    min_L:    " + format(float(str(np.mean(min_ang_ankle_l_np))), '.1f') + "    (min_R+min_L)/2:    " + format(float(str((np.mean(min_ang_ankle_r_np) + np.mean(min_ang_ankle_l_np)) / 2)), '.1f') + "    min_R/min_L:    " + format(float(str(np.mean(min_ang_ankle_r_np) / np.mean(min_ang_ankle_l_np))), '.1f'))
    else:
        angle_folding_lb = ttk.Label(frame_calculations, text="Угол разгибания плеча [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_folding_lb.place(relx=0.03, rely=0.36)

    if check_parameter_18.get():
        angle_hip_extension_lb = ttk.Label(frame_calculations, text="Угол выноса бедра [град.]:\nmin_R:    " + '---' + "    min_L:    " + '---' + "    (min_R+min_L)/2:    " + '---' + "    min_R/min_L:    " + '---')
    else:
        angle_hip_extension_lb = ttk.Label(frame_calculations, text="Угол выноса бедра [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_hip_extension_lb.place(relx=0.03, rely=0.41)

    if check_parameter_19.get():
        angle_hip_dop_lb = ttk.Label(frame_calculations, text="Угол сведения бедер и фазовый путь бедра:       ---")
    else:
        angle_hip_dop_lb = ttk.Label(frame_calculations, text="Угол сведения бедер и фазовый путь бедра:       ---")
    angle_hip_dop_lb.place(relx=0.03, rely=0.46)


    # Движение опорной ноги при постановке на опору --------------------------------------------------------------------------------
    global ang_knee_fs_l_np, ang_knee_fs_r_np, ang_hip_fs_l_np, ang_hip_fs_r_np, takeaway_step_fs_l_np, takeaway_step_fs_r_np
    global ang_ankle_fs_l_np, ang_ankle_fs_r_np, ang_bed_vert_l_np, ang_bed_vert_r_np
    global ang_gol_op_fs_l_np, ang_gol_op_fs_r_np
    
    ang_knee_fs_l = []
    ang_knee_fs_r = []
    ang_hip_fs_l = []
    ang_hip_fs_r = []
    takeaway_step_fs_l = []
    takeaway_step_fs_r = []
    ang_ankle_fs_l = []
    ang_ankle_fs_r = []
    ang_bed_vert_l = []
    ang_bed_vert_r = []
    ang_gol_op_fs_l = []
    ang_gol_op_fs_r = []
    
    for i in range(1, m.num_steps + 1):
        
        ang_ankle_fs_l += [m.angle(Knee_L, Ankle_L, Toe_L, event=fs + '_L' + str(i))]
        ang_ankle_fs_r += [m.angle(Knee_R, Ankle_R, Toe_R, event=fs + '_R' + str(i))]
        
        ang_bed_vert_l += [m.angle(Knee_L, Hip_L, '000', '100', event=fs+'_L'+str(i))]
        ang_bed_vert_r += [m.angle(Knee_R, Hip_R, '000', '100', event=fs+'_R'+str(i))]
        
        ang_gol_op_fs_l += [m.angle('000', '001', Ankle_L, Knee_L, event = fs + '_L' + str(i))]
        ang_gol_op_fs_r += [m.angle('000', '001', Ankle_R, Knee_R, event = fs + '_R' + str(i))]
        
        ang_knee_fs_l += [m.angle(Hip_L, Knee_L, Ankle_L, event=fs + '_L' + str(i))]
        ang_knee_fs_r += [m.angle(Hip_R, Knee_R, Ankle_R, event=fs + '_R' + str(i))]
        ang_hip_fs_l += [m.angle(Knee_L, Hip_L, Back_B, Back_T, event=fs + '_L' + str(i))]
        ang_hip_fs_r += [m.angle(Knee_R, Hip_R, Back_B, Back_T, event=fs + '_R' + str(i))]
        if direction == '-x':
            d = m.distance(Heel_L, Hip_L, event=fs + '_L' + str(i))
            alpha = m.angle(Heel_L, Hip_L, '000', '001', event=fs + '_L' + str(i))
            takeaway_step_fs_l += [d * math.cos(alpha * 3.1416 / 180)]
            d = m.distance(Heel_R, Hip_R, event=fs + '_R' + str(i))
            alpha = m.angle(Heel_R, Hip_R, '000', '001', event=fs + '_R' + str(i))
            takeaway_step_fs_r += [d * math.cos(alpha * 3.1416 / 180)]

            
    ang_gol_op_fs_l_np = np.array(ang_gol_op_fs_l[steps_from:steps_to]) + 90 - gol_op_left_st
    ang_gol_op_fs_r_np = np.array(ang_gol_op_fs_r[steps_from:steps_to]) + 90 - gol_op_right_st
    
    ang_bed_vert_l_np = np.array(ang_bed_vert_l[steps_from:steps_to]) + bed_vert_left_st
    ang_bed_vert_r_np = np.array(ang_bed_vert_r[steps_from:steps_to]) + bed_vert_right_st
    ang_ankle_fs_l_np = np.array(ang_ankle_fs_l[steps_from:steps_to]) - left_gol_st
    ang_ankle_fs_r_np = np.array(ang_ankle_fs_r[steps_from:steps_to]) - right_gol_st
    ang_knee_fs_l_np = np.array(ang_knee_fs_l[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_L)
    ang_knee_fs_r_np = np.array(ang_knee_fs_r[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_R)
    ang_hip_fs_l_np = np.array(ang_hip_fs_l[steps_from:steps_to]) - korp_bed_left_st
    ang_hip_fs_r_np = np.array(ang_hip_fs_r[steps_from:steps_to]) - korp_bed_right_st
    takeaway_step_fs_l_np = np.array(takeaway_step_fs_l[steps_from:steps_to]) / 10
    takeaway_step_fs_r_np = np.array(takeaway_step_fs_r[steps_from:steps_to]) / 10

    if check_parameter_20.get():
        knee_angle_lb = ttk.Label(frame_calculations, text="Угол в коленном суставе [град.]:\nR:    " + format(float(str(np.mean(ang_knee_fs_r_np))), '.1f') + "    L:    " + format(float(str(np.mean(ang_knee_fs_l_np))), '.1f') + "    (R + L)/2:    " + format(float(str(np.mean(ang_knee_fs_r_np) / 2 + np.mean(ang_knee_fs_l_np) / 2)), '.1f') + "    R/L:    " + format(float(str(np.mean(ang_knee_fs_r_np) / np.mean(ang_knee_fs_l_np))), '.1f'))
    else:
        knee_angle_lb = ttk.Label(frame_calculations, text="Угол в коленном суставе [град.]:\nR:    ---    L:    ---(R + L)/2    :    ---R/L    :    ---")
    knee_angle_lb.place(relx=0.35, rely=0.36)

    if check_parameter_21.get():
        pelvic_angle_lb = ttk.Label(frame_calculations, text="Угол в тазобедренном суставе [град.]:\nR:    " + format(float(str(np.mean(ang_hip_fs_r_np))), '.1f') + "    L:    " + format(float(str(np.mean(ang_hip_fs_l_np))), '.1f') + "    (R + L)/2:    " + format(float(str(np.mean(ang_hip_fs_r_np) / 2 + np.mean(ang_hip_fs_l_np) / 2)), '.1f') + "    R/L:    " + format(float(str(np.mean(ang_hip_fs_r_np) / np.mean(ang_hip_fs_l_np))), '.1f'))
    else:
        pelvic_angle_lb = ttk.Label(frame_calculations, text="Угол в тазобедренном суставе [град.]:\nR:    ---    L:    ---(R + L)/2    :    ---R/L    :    ---")
    pelvic_angle_lb.place(relx=0.35, rely=0.41)

    if check_parameter_22.get():
        foot_strike_lb = ttk.Label(frame_calculations, text="Вынос стопы  [см]:\nR:    " + format(float(str(np.mean(takeaway_step_fs_r_np))), '.1f') + "    L:    " + format(float(str(np.mean(takeaway_step_fs_l_np))), '.1f') + "    (R + L)/2:    " + format(float(str(np.mean(takeaway_step_fs_r_np) / 2 + np.mean(takeaway_step_fs_l_np) / 2)), '.1f') + "    R/L:    " + format(float(str(np.mean(takeaway_step_fs_r_np) / np.mean(takeaway_step_fs_l_np))), '.1f'))
    else:
        foot_strike_lb = ttk.Label(frame_calculations, text="Вынос стопы  [см]:\nR:    ---    L:    ---(R + L)/2    :    ---R/L    :    ---")
    foot_strike_lb.place(relx=0.35, rely=0.46)

    # Движение опорной ноги при прохождении вертикали --------------------------------------------------------------------------------
    global ang_knee_ms_l_np, ang_knee_ms_r_np, ang_hip_ms_l_np, ang_hip_ms_r_np, heel_height_ms_l_np, heel_height_ms_r_np

    ang_knee_ms_l = []
    ang_knee_ms_r = []
    ang_hip_ms_l = []
    ang_hip_ms_r = []
    heel_height_ms_l = []
    heel_height_ms_r = []
    for i in range(1, m.num_steps + 1):
        ang_knee_ms_l += [m.angle(Hip_L, Knee_L, Ankle_L, event=ms + '_L' + str(i))]
        ang_knee_ms_r += [m.angle(Hip_R, Knee_R, Ankle_R, event=ms + '_R' + str(i))]
        ang_hip_ms_l += [m.angle(Knee_L, Hip_L, Back_B, Back_T, event=ms + '_L' + str(i))]
        ang_hip_ms_r += [m.angle(Knee_R, Hip_R, Back_B, Back_T, event=ms + '_R' + str(i))]
        heel_height_ms_l += [m._get_point(Heel_L, m.frame(event=ms + '_L' + str(i))).Z - zero_z_L]
        heel_height_ms_r += [m._get_point(Heel_R, m.frame(event=ms + '_R' + str(i))).Z - zero_z_R]

    ang_knee_ms_l_np = np.array(ang_knee_ms_l[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_L)
    ang_knee_ms_r_np = np.array(ang_knee_ms_r[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_R)
    ang_hip_ms_l_np = np.array(ang_hip_ms_l[steps_from:steps_to]) - korp_bed_left_st
    ang_hip_ms_r_np = np.array(ang_hip_ms_r[steps_from:steps_to]) - korp_bed_right_st
    heel_height_ms_l_np = np.array(heel_height_ms_l[steps_from:steps_to]) / 10
    heel_height_ms_r_np = np.array(heel_height_ms_r[steps_from:steps_to]) / 10

    if check_parameter_21.get():
        angle_knee_lb = ttk.Label(frame_calculations, text="Угол в коленном суставе [град.]:\nmin_R:    " + format(float(str(np.mean(ang_knee_ms_r_np))), '.1f') + "    min_L:    " + format(float(str(np.mean(ang_knee_ms_l_np))), '.1f') + "    (min_R+min_L)/2:    " + format(float(str(np.mean(ang_knee_ms_r_np) / 2 + np.mean(ang_knee_ms_l_np) / 2)), '.1f') + "    min_R/min_L:    " + format(float(str(np.mean(ang_knee_ms_r_np) / np.mean(ang_knee_ms_l_np))), '.1f'))
    else:
        angle_knee_lb = ttk.Label(frame_calculations, text="Угол в коленном суставе [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_knee_lb.place(relx=0.6, rely=0.41)

    if check_parameter_22.get():
        angle_hip_lb = ttk.Label(frame_calculations, text="Угол в тазобедренном суставе [град.]:\nmin_R:    " + format(float(str(np.mean(ang_hip_ms_r_np))), '.1f') + "    min_L:    " + format(float(str(np.mean(ang_hip_ms_l_np))), '.1f') + "    (min_R+min_L)/2:    " + format(float(str(np.mean(ang_hip_ms_r_np) / 2 + np.mean(ang_hip_ms_l_np) / 2)), '.1f') + "    min_R/min_L:    " + format(float(str(np.mean(ang_hip_ms_r_np) / np.mean(ang_hip_ms_l_np))), '.1f'))
    else:
        angle_hip_lb = ttk.Label(frame_calculations, text="Угол в тазобедренном суставе [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_hip_lb.place(relx=0.6, rely=0.46)

    if check_parameter_23.get():
        heel_height_lb = ttk.Label(frame_calculations, text="Высота пятки [см]:\nR:    " + '---' + "    L:    " + '---' + "    (R + L)/2:    " + '---' + "    R/L:    " + '---')
    else:
        heel_height_lb = ttk.Label(frame_calculations, text="Высота пятки [см]:\nR:    ---    L:    ---(R + L)/2    :    ---R/L    :    ---")
    heel_height_lb.place(relx=0.6, rely=0.51)

    # Движение опорной ноги при завершении отталкивания
    global ang_knee_to_l_np, ang_knee_to_r_np, ang_ankle_to_l_np, ang_ankle_to_r_np, ang_hip_vert_to_l_np, ang_hip_vert_to_r_np
    global ang_hip_back_to_l_np, ang_hip_back_to_r_np, ang_gol_op_to_l_np, ang_gol_op_to_r_np
    
    ang_knee_to_l = []
    ang_knee_to_r = []
    ang_ankle_to_l = []
    ang_ankle_to_r = []
    ang_hip_vert_to_l = []
    ang_hip_vert_to_r = []
    ang_hip_back_to_l = []
    ang_hip_back_to_r = []
    ang_gol_op_to_l = []
    ang_gol_op_to_r = []
    for i in range(1, m.num_steps + 1):
        ang_knee_to_l += [m.angle(Hip_L, Knee_L, Ankle_L, event=to + '_L' + str(i))]
        ang_knee_to_r += [m.angle(Hip_R, Knee_R, Ankle_R, event=to + '_R' + str(i))]
        ang_ankle_to_l += [m.angle(Knee_L, Ankle_L, Toe_L, event=to + '_L' + str(i))]
        ang_ankle_to_r += [m.angle(Knee_R, Ankle_R, Toe_R, event=to + '_R' + str(i))]
        ang_hip_vert_to_l += [m.angle(Knee_L, Hip_L, '000', '100', event=to + '_L' + str(i))]
        ang_hip_vert_to_r += [m.angle(Knee_R, Hip_R, '000', '100', event=to + '_R' + str(i))]
        ang_hip_back_to_l += [m.angle(Back_T, Back_B, Hip_L, Knee_L, event =to + '_L' + str(i))]
        ang_hip_back_to_r += [m.angle(Back_T, Back_B, Hip_R, Knee_R, event =to + '_R' + str(i))]
        ang_gol_op_to_l += [m.angle('000', '001', Ankle_L, Knee_L, event = to + '_L' + str(i))]
        ang_gol_op_to_r += [m.angle('000', '001', Ankle_R, Knee_R, event = to + '_R' + str(i))]
        
    ang_knee_to_l_np = np.array(ang_knee_to_l[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_L)
    ang_knee_to_r_np = np.array(ang_knee_to_r[steps_from:steps_to]) + (180 - stat_Hip_Knee_Ankle_R)
    ang_ankle_to_l_np = np.array(ang_ankle_to_l[steps_from:steps_to]) - left_gol_st
    ang_ankle_to_r_np = np.array(ang_ankle_to_r[steps_from:steps_to]) - right_gol_st
    ang_hip_vert_to_l_np = np.array(ang_hip_vert_to_l[steps_from:steps_to]) - bed_vert_left_st
    ang_hip_vert_to_r_np = np.array(ang_hip_vert_to_r[steps_from:steps_to]) - bed_vert_right_st
    ang_hip_back_to_l_np = np.array(ang_hip_back_to_l[steps_from:steps_to]) - korp_bed_left_st
    ang_hip_back_to_r_np = np.array(ang_hip_back_to_r[steps_from:steps_to]) - korp_bed_right_st
    ang_gol_op_to_l_np = np.array(ang_gol_op_to_l[steps_from:steps_to]) + 90 - gol_op_left_st
    ang_gol_op_to_r_np = np.array(ang_gol_op_to_r[steps_from:steps_to]) + 90 - gol_op_right_st
    #print(korp_bed_left_st, korp_bed_right_st)

    if check_parameter_24.get():
        angle_knee_lb = ttk.Label(frame_calculations, text="Угол в коленном суставе [град.]:\nmin_R:    " + format(float(str(np.mean(ang_knee_to_r_np))), '.1f') + "    min_L:    " + format(float(str(np.mean(ang_knee_to_l_np))), '.1f') + "    (min_R+min_L)/2:    " + format(float(str(np.mean(ang_knee_to_r_np) / 2 + np.mean(ang_knee_to_l_np) / 2)), '.1f') + "    min_R/min_L:    " + format(float(str(np.mean(ang_knee_to_r_np) / np.mean(ang_knee_to_l_np))), '.1f'))
    else:
        angle_knee_lb = ttk.Label(frame_calculations, text="Угол в коленном суставе [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_knee_lb.place(relx=0.03, rely=0.56)

    if check_parameter_25.get():
        angle_galenostop_lb = ttk.Label(frame_calculations, text="Угол в голеностопном суставе [град.]:\nmin_R:    " + format(float(str(np.mean(ang_ankle_to_r_np))), '.1f') + "    min_L:    " + format(float(str(np.mean(ang_ankle_to_l_np))), '.1f') + "    (min_R+min_L)/2:    " + format(float(str(np.mean(ang_ankle_to_r_np) / 2 + np.mean(ang_ankle_to_l_np) / 2)), '.1f') + "    min_R/min_L:    " + format(float(str(np.mean(ang_ankle_to_r_np) / np.mean(ang_ankle_to_l_np))), '.1f'))
    else:
        angle_galenostop_lb = ttk.Label(frame_calculations, text="Угол в голеностопном суставе [град.]:\nmin_R:      ---      min_L:      ---      (min_R+min_L)/2:      ---      min_R/min_L:      ---")
    angle_galenostop_lb.place(relx=0.03, rely=0.61)

    if check_parameter_26.get():
        hip_angle_with_vertical_lb = ttk.Label(frame_calculations, text="Угол бедра с вертикалью [град.]:\nmin_R:    " + format(float(str(np.mean(ang_hip_vert_to_r_np))), '.1f') + "    min_L:    " + format(float(str(np.mean(ang_hip_vert_to_l_np))), '.1f') + "    (min_R + min_L)/2:    " + format(float(str(np.mean(ang_hip_vert_to_r_np) / 2 + np.mean(ang_hip_vert_to_l_np) / 2)), '.1f') + "    min_R/min_L:    " + format(float(str(np.mean(ang_hip_vert_to_r_np) / np.mean(ang_hip_vert_to_l_np))), '.1f'))
    else:
        hip_angle_with_vertical_lb = ttk.Label(frame_calculations, text="Угол бедра с вертикалью [град.]:\nmin_R:    ---    min_L:    ---(min_R + min_L)/2    :    ---min_R/min_L    :    ---")
    hip_angle_with_vertical_lb.place(relx=0.03, rely=0.66)

    def graf():
        if type_move.get() == 'Бег' and (np.mean(fly_phase_L) <= 0 or np.mean(fly_phase_R) <= 0):
            showinfo(title="Информация", message="Отсутствует фаза полета!", icon=INFO, default=OK)
            return
        fig = plt.figure(figsize=(15, 5))
        offset = .3
        if type_move.get() == 'Бег':
            data = np.array([np.mean(fly_phase_L), np.mean(depreciation_phase_L), np.mean(repulsion_phase_L)])
            labels = ['Фаза полета', 'Фаза амортизации', 'Фаза отталкивания']
        elif type_move.get() == 'Ходьба':
            data = np.array([np.mean(single_contact_phase_R), np.mean(depreciation_phase_L), np.mean(repulsion_phase_L)])
            labels = ['Фаза переноса', 'Фаза амортизации', 'Фаза отталкивания']
        explode = (0.1, 0.15, 0.15)
        ax = fig.add_subplot(121)
        ax.pie(data.flatten(), radius=1.5 - offset, wedgeprops=dict(width=offset, edgecolor='w'), explode=explode,
               shadow=True, autopct='%1.1f%%', labels=labels, rotatelabels=False)
        time_L = np.mean(fly_phase_L) + np.mean(depreciation_phase_L) + np.mean(repulsion_phase_L)
        k = 2
        if type_move.get() == 'Бег':
            k = 1
        ax.text(x=-.2, y=0, s=f"{round(time_L*k, 2)} сек")
        ax.set_title('Левая нога')
        ax2 = fig.add_subplot(122)
        if type_move.get() == 'Бег':
            data = np.array([np.mean(fly_phase_R), np.mean(depreciation_phase_R), np.mean(repulsion_phase_R)])
        elif type_move.get() == 'Ходьба':
            data = np.array([np.mean(single_contact_phase_L), np.mean(depreciation_phase_R), np.mean(repulsion_phase_R)])
        ax2.pie(data.flatten(), radius=1.5 - offset, wedgeprops=dict(width=offset, edgecolor='w'), explode=explode,
                shadow=True, autopct='%1.1f%%', labels=labels, rotatelabels=False)
        time_R = np.mean(fly_phase_R) + np.mean(depreciation_phase_R) + np.mean(repulsion_phase_R)
        ax2.text(x=-.2, y=0, s=f"{round(time_R*k, 2)} сек")
        ax2.set_title('Правая нога')
        plt.show()

    btn_gr = ttk.Button(frame_calculations, text="Вывести график", command=graf)
    btn_gr.place(relx=0.09, rely=0.25)

    zagalovok1 = ttk.Label(frame_calculations, text="Параметры бегового шага:")
    zagalovok1.place(relx=0.06, rely=0.02)

    zagalovok2 = ttk.Label(frame_calculations, text="Движение корпуса:")
    zagalovok2.place(relx=0.33, rely=0.02)

    zagalovok3 = ttk.Label(frame_calculations, text="Движение рук:")
    zagalovok3.place(relx=0.71, rely=0.02)

    zagalovok4 = ttk.Label(frame_calculations, text="Движение маховой ноги:")
    zagalovok4.place(relx=0.11, rely=0.32)

    zagalovok5 = ttk.Label(frame_calculations, text="Движение опорной ноги при постановке на опору:")
    zagalovok5.place(relx=0.34, rely=0.32)

    zagalovok6 = ttk.Label(frame_calculations, text="Движение опорной ноги при прохождении вертикали:")
    zagalovok6.place(relx=0.64, rely=0.37)

    zagalovok7 = ttk.Label(frame_calculations, text="Движение опорной ноги при завершении отталкивания:")
    zagalovok7.place(relx=0.09, rely=0.52)

    # Функции для вывода графиков по маркерам

    def steps_from_graf_selected_graphics(*args):
        if (steps_to_graf.get() < steps_from_graf.get()):
            steps_to_graf["textvariable"] = steps_from_graf.get()
        steps_to_graf["value"] = list(range(int(steps_from_graf.get()), m.num_steps))


    def show_graphic(data, axis):
        global window
        window = Tk()
        window.title("Выберите промежуток шагов")
        window.geometry("350x220+600+350")
        global steps_from_graf, steps_to_graf
        steps_from_graf = ttk.Combobox(window, width=8)
        steps_to_graf = ttk.Combobox(window, width=8)
        steps = list(range(1, m.num_steps))
        steps_from_graf["value"] = steps
        steps_from_cb.set('')
        steps_from_graf.insert(0, steps[0])
        steps_to_graf["value"] = steps
        steps_to_cb.set('')
        steps_to_graf.insert(0, steps[-1])

        superimposing_lb = ttk.Label(window, text="Отображение графиков с наложением шагов")
        superimposing_lb.place(relx=0.13, rely=0.44)

        superimposing_cb = ttk.Combobox(window, values=["Да", "Нет"], width=8)
        superimposing_cb.place(relx=0.4, rely=0.56)
        superimposing_cb.set("Нет")


        lb_from = ttk.Label(window, text="C")
        lb_to = ttk.Label(window, text="ПО")
        lb_from.place(relx=0.28, rely=0.15)
        lb_to.place(relx=0.67, rely=0.15)

        steps_from_graf.place(relx=0.2, rely=0.27)
        steps_to_graf.place(relx=0.6, rely=0.27)

        steps_from_graf.bind("<<ComboboxSelected>>", steps_from_graf_selected_graphics)

        def return_graf():
            if superimposing_cb.get() == "Да":
                return data_graphic(int(steps_from_graf.get()), int(steps_to_graf.get()), data, axis)  # from and to steps, name data
            else:
                return data_graphic_1(int(steps_from_graf.get()), int(steps_to_graf.get()), data, axis)

        save_steps_to_graf = ttk.Button(window, text="Вывести график", command=return_graf)
        save_steps_to_graf.place(relx=0.37, rely=0.8)


    def data_graphic_1(st_from, st_to, data, axis):
        window.destroy()
        #print(st_from, st_to, data, axis)
        mas_x = []
        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]

        x = np.linspace(0, (st_to - st_from + 1) * 100, mas_x[-1]-mas_x[0])
        mas_y = []

        for i in range(mas_x[0], mas_x[-1]):
            mas_y.append(m.__data__[data].loc[i])
        y = mas_y
        plt.figure(data + " from " + str(st_from) + " to " + str(st_to) + " steps")
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        plt.ylabel(f"мм  (ось  {axis})")
        plt.grid()
        plt.plot(x, y)
        plt.show()
        
    def data_graphic(st_from, st_to, data, axis):
        window.destroy()
        mas_x = []
        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]

        mas_y = [[] for i in range(st_from, st_to + 2)]

        for k in range(st_from, st_to + 1):
            for i in range(mas_x[k - st_from], mas_x[k+1-st_from]):
                mas_y[k - st_from] += [m.__data__[data].loc[i]]

        y = mas_y
        plt.figure(data + " from " + str(st_from) + " to " + str(st_to) + " steps")
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        plt.ylabel(f"мм  (ось  {axis})")
        plt.grid()
        index = 0
        for i in y:
            x = np.linspace(0, 100, len(i))
            if index + st_from > st_to:
                continue
            plt.plot(x, i, label=f'{st_from + index} шаг')
            index += 1
        plt.legend()
        plt.show()


    def Back_T_pkm(event):
        x = event.x
        y = event.y
        Back_T_mn.post(event.x_root, event.y_root)


    # Back_T_graphics -------------------------------------------------------------------------------

    Back_T_mn = Menu(tearoff=0)
    Back_T_mn.add_command(label="3D X-position", command=lambda: show_graphic(Back_T + " X", 'x'))
    Back_T_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Back_T + " Y", 'y'))
    Back_T_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Back_T + " Z", 'z'))
    Back_T_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Back_T, "x"))
    Back_T_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Back_T, "y"))
    Back_T_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Back_T, "z"))
    Back_T_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Back_T))

    Back_T_lb = Label(frame_graphics, text=Back_T, background="white", width=13, font=("Arial Bold", 14),
                      highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Back_T_lb.place(relx=-0.001, rely=0.05)

    Back_T_lb.bind('<Button-3>', Back_T_pkm)


    # Back_B_graphics -------------------------------------------------------------------------------

    def Back_B_pkm(event):
        x = event.x
        y = event.y
        Back_B_mn.post(event.x_root, event.y_root)

    Back_B_mn = Menu(tearoff=0)
    Back_B_mn.add_command(label="3D X-position", command=lambda: show_graphic(Back_B + " X", 'x'))
    Back_B_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Back_B + " Y", 'y'))
    Back_B_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Back_B + " Z", 'z'))
    Back_B_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Back_B, "x"))
    Back_B_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Back_B, "y"))
    Back_B_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Back_B, "z"))
    Back_B_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Back_B))

    Back_B_lb = Label(frame_graphics, text=Back_B, background="white", width=13, font=("Arial Bold", 14),
                      highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Back_B_lb.place(relx=-0.001, rely=0.1)

    Back_B_lb.bind('<Button-3>', Back_B_pkm)


    # Shoulder_R_graphics -------------------------------------------------------------------------------

    def Shoulder_R_pkm(event):
        x = event.x
        y = event.y
        Shoulder_R_mn.post(event.x_root, event.y_root)

    Shoulder_R_mn = Menu(tearoff=0)
    Shoulder_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Shoulder_R + " X", 'x'))
    Shoulder_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Shoulder_R + " Y", 'y'))
    Shoulder_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Shoulder_R + " Z", 'z'))
    Shoulder_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Shoulder_R, "x"))
    Shoulder_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Shoulder_R, "y"))
    Shoulder_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Shoulder_R, "z"))
    Shoulder_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Shoulder_R))

    Shoulder_R_lb = Label(frame_graphics, text=Shoulder_R, background="white", width=13, font=("Arial Bold", 14),
                          highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Shoulder_R_lb.place(relx=-0.001, rely=0.15)

    Shoulder_R_lb.bind('<Button-3>', Shoulder_R_pkm)


    # Shoulder_L_graphics -------------------------------------------------------------------------------

    def Shoulder_L_pkm(event):
        x = event.x
        y = event.y
        Shoulder_L_mn.post(event.x_root, event.y_root)

    Shoulder_L_mn = Menu(tearoff=0)
    Shoulder_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Shoulder_L + " X", 'x'))
    Shoulder_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Shoulder_L + " Y", 'y'))
    Shoulder_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Shoulder_L + " Z", 'z'))
    Shoulder_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Shoulder_L, "x"))
    Shoulder_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Shoulder_L, "y"))
    Shoulder_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Shoulder_L, "z"))
    Shoulder_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Shoulder_L))

    Shoulder_L_lb = Label(frame_graphics, text=Shoulder_L, background="white", width=13, font=("Arial Bold", 14),
                          highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Shoulder_L_lb.place(relx=-0.001, rely=0.2)

    Shoulder_L_lb.bind('<Button-3>', Shoulder_L_pkm)


    # Elbow_R_graphics -------------------------------------------------------------------------------

    def Elbow_R_pkm(event):
        x = event.x
        y = event.y
        Elbow_R_mn.post(event.x_root, event.y_root)

    Elbow_R_mn = Menu(tearoff=0)
    Elbow_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Elbow_R + " X", 'x'))
    Elbow_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Elbow_R + " Y", 'y'))
    Elbow_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Elbow_R + " Z", 'z'))
    Elbow_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Elbow_R, "x"))
    Elbow_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Elbow_R, "y"))
    Elbow_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Elbow_R, "z"))
    Elbow_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Elbow_R))

    Elbow_R_lb = Label(frame_graphics, text=Elbow_R, background="white", width=13, font=("Arial Bold", 14),
                       highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Elbow_R_lb.place(relx=-0.001, rely=0.25)

    Elbow_R_lb.bind('<Button-3>', Elbow_R_pkm)


    # Elbow_L_graphics -------------------------------------------------------------------------------

    def Elbow_L_pkm(event):
        x = event.x
        y = event.y
        Elbow_L_mn.post(event.x_root, event.y_root)

    Elbow_L_mn = Menu(tearoff=0)
    Elbow_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Elbow_L + " X", 'x'))
    Elbow_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Elbow_L + " Y", 'y'))
    Elbow_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Elbow_L + " Z", 'z'))
    Elbow_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Elbow_L, "x"))
    Elbow_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Elbow_L, "y"))
    Elbow_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Elbow_L, "z"))
    Elbow_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Elbow_L))

    Elbow_L_lb = Label(frame_graphics, text=Elbow_L, background="white", width=13, font=("Arial Bold", 14),
                       highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Elbow_L_lb.place(relx=-0.001, rely=0.3)

    Elbow_L_lb.bind('<Button-3>', Elbow_L_pkm)


    # Wrist_R_graphics -------------------------------------------------------------------------------

    def Wrist_R_pkm(event):
        x = event.x
        y = event.y
        Wrist_R_mn.post(event.x_root, event.y_root)

    Wrist_R_mn = Menu(tearoff=0)
    Wrist_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Wrist_R + " X", 'x'))
    Wrist_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Wrist_R + " Y", 'y'))
    Wrist_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Wrist_R + " Z", 'z'))
    Wrist_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Wrist_R, "x"))
    Wrist_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Wrist_R, "y"))
    Wrist_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Wrist_R, "z"))
    Wrist_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Wrist_R))

    Wrist_R_lb = Label(frame_graphics, text=Wrist_R, background="white", width=13, font=("Arial Bold", 14),
                       highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Wrist_R_lb.place(relx=-0.001, rely=0.35)

    Wrist_R_lb.bind('<Button-3>', Wrist_R_pkm)


    # Wrist_L_graphics -------------------------------------------------------------------------------

    def Wrist_L_pkm(event):
        x = event.x
        y = event.y
        Wrist_L_mn.post(event.x_root, event.y_root)

    Wrist_L_mn = Menu(tearoff=0)
    Wrist_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Wrist_L + " X", 'x'))
    Wrist_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Wrist_L + " Y", 'y'))
    Wrist_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Wrist_L + " Z", 'z'))
    Wrist_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Wrist_L, "x"))
    Wrist_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Wrist_L, "y"))
    Wrist_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Wrist_L, "z"))
    Wrist_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Wrist_L))

    Wrist_L_lb = Label(frame_graphics, text=Wrist_L, background="white", width=13, font=("Arial Bold", 14),
                       highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Wrist_L_lb.place(relx=-0.001, rely=0.4)

    Wrist_L_lb.bind('<Button-3>', Wrist_L_pkm)


    # Hip_R_graphics -------------------------------------------------------------------------------

    def Hip_R_pkm(event):
        x = event.x
        y = event.y
        Hip_R_mn.post(event.x_root, event.y_root)

    Hip_R_mn = Menu(tearoff=0)
    Hip_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Hip_R + " X", 'x'))
    Hip_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Hip_R + " Y", 'y'))
    Hip_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Hip_R + " Z", 'z'))
    Hip_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Hip_R, "x"))
    Hip_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Hip_R, "y"))
    Hip_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Hip_R, "z"))
    Hip_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Hip_R))

    Hip_R_lb = Label(frame_graphics, text=Hip_R, background="white", width=13, font=("Arial Bold", 14),
                     highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Hip_R_lb.place(relx=-0.001, rely=0.45)

    Hip_R_lb.bind('<Button-3>', Hip_R_pkm)


    # Hip_L_graphics -------------------------------------------------------------------------------

    def Hip_L_pkm(event):
        x = event.x
        y = event.y
        Hip_L_mn.post(event.x_root, event.y_root)

    Hip_L_mn = Menu(tearoff=0)
    Hip_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Hip_L + " X", 'x'))
    Hip_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Hip_L + " Y", 'y'))
    Hip_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Hip_L + " Z", 'z'))
    Hip_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Hip_L, "x"))
    Hip_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Hip_L, "y"))
    Hip_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Hip_L, "z"))
    Hip_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Hip_L))

    Hip_L_lb = Label(frame_graphics, text=Hip_L, background="white", width=13, font=("Arial Bold", 14),
                     highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Hip_L_lb.place(relx=-0.001, rely=0.5)

    Hip_L_lb.bind('<Button-3>', Hip_L_pkm)


    # Knee_R_graphics -------------------------------------------------------------------------------

    def Knee_R_pkm(event):
        x = event.x
        y = event.y
        Knee_R_mn.post(event.x_root, event.y_root)

    Knee_R_mn = Menu(tearoff=0)
    Knee_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Knee_R + " X", 'x'))
    Knee_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Knee_R + " Y", 'y'))
    Knee_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Knee_R + " Z", 'z'))
    Knee_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Knee_R, "x"))
    Knee_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Knee_R, "y"))
    Knee_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Knee_R, "z"))
    Knee_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Knee_R))

    Knee_R_lb = Label(frame_graphics, text=Knee_R, background="white", width=13, font=("Arial Bold", 14),
                      highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Knee_R_lb.place(relx=-0.001, rely=0.55)

    Knee_R_lb.bind('<Button-3>', Knee_R_pkm)


    # Knee_L_graphics -------------------------------------------------------------------------------

    def Knee_L_pkm(event):
        x = event.x
        y = event.y
        Knee_L_mn.post(event.x_root, event.y_root)

    Knee_L_mn = Menu(tearoff=0)
    Knee_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Knee_L + " X", 'x'))
    Knee_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Knee_L + " Y", 'y'))
    Knee_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Knee_L + " Z", 'z'))
    Knee_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Knee_L, "x"))
    Knee_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Knee_L, "y"))
    Knee_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Knee_L, "z"))
    Knee_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Knee_L))


    Knee_L_lb = Label(frame_graphics, text=Knee_L, background="white", width=13, font=("Arial Bold", 14),
                      highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Knee_L_lb.place(relx=-0.001, rely=0.6)

    Knee_L_lb.bind('<Button-3>', Knee_L_pkm)


    # Ankle_R_graphics -------------------------------------------------------------------------------

    def Ankle_R_pkm(event):
        x = event.x
        y = event.y
        Ankle_R_mn.post(event.x_root, event.y_root)

    Ankle_R_mn = Menu(tearoff=0)
    Ankle_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Ankle_R + " X", 'x'))
    Ankle_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Ankle_R + " Y", 'y'))
    Ankle_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Ankle_R + " Z", 'z'))
    Ankle_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Ankle_R, "x"))
    Ankle_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Ankle_R, "y"))
    Ankle_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Ankle_R, "z"))
    Ankle_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Ankle_R))


    Ankle_R_lb = Label(frame_graphics, text=Ankle_R, background="white", width=13, font=("Arial Bold", 14),
                       highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Ankle_R_lb.place(relx=-0.001, rely=0.65)

    Ankle_R_lb.bind('<Button-3>', Ankle_R_pkm)


    # Ankle_L_graphics -------------------------------------------------------------------------------

    def Ankle_L_pkm(event):
        x = event.x
        y = event.y
        Ankle_L_mn.post(event.x_root, event.y_root)

    Ankle_L_mn = Menu(tearoff=0)
    Ankle_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Ankle_L + " X", 'x'))
    Ankle_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Ankle_L + " Y", 'y'))
    Ankle_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Ankle_L + " Z", 'z'))
    Ankle_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Ankle_L, "x"))
    Ankle_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Ankle_L, "y"))
    Ankle_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Ankle_L, "z"))
    Ankle_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Ankle_L))


    Ankle_L_lb = Label(frame_graphics, text=Ankle_L, background="white", width=13, font=("Arial Bold", 14),
                       highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Ankle_L_lb.place(relx=-0.001, rely=0.7)

    Ankle_L_lb.bind('<Button-3>', Ankle_L_pkm)


    # Heel_R_graphics -------------------------------------------------------------------------------

    def Heel_R_pkm(event):
        x = event.x
        y = event.y
        Heel_R_mn.post(event.x_root, event.y_root)

    Heel_R_mn = Menu(tearoff=0)
    Heel_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Heel_R + " X", 'x'))
    Heel_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Heel_R + " Y", 'y'))
    Heel_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Heel_R + " Z", 'z'))
    Heel_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Heel_R, "x"))
    Heel_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Heel_R, "y"))
    Heel_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Heel_R, "z"))
    Heel_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Heel_R))


    Heel_R_lb = Label(frame_graphics, text=Heel_R, background="white", width=13, font=("Arial Bold", 14),
                      highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Heel_R_lb.place(relx=-0.001, rely=0.75)

    Heel_R_lb.bind('<Button-3>', Heel_R_pkm)


    # Heel_L_graphics -------------------------------------------------------------------------------

    def Heel_L_pkm(event):
        x = event.x
        y = event.y
        Heel_L_mn.post(event.x_root, event.y_root)

    Heel_L_mn = Menu(tearoff=0)
    Heel_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Heel_L + " X", 'x'))
    Heel_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Heel_L + " Y", 'y'))
    Heel_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Heel_L + " Z", 'z'))
    Heel_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Heel_L, "x"))
    Heel_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Heel_L, "y"))
    Heel_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Heel_L, "z"))
    Heel_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Heel_L))


    Heel_L_lb = Label(frame_graphics, text=Heel_L, background="white", width=13, font=("Arial Bold", 14),
                      highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Heel_L_lb.place(relx=-0.001, rely=0.8)

    Heel_L_lb.bind('<Button-3>', Heel_L_pkm)


    # Toe_R_graphics -------------------------------------------------------------------------------

    def Toe_R_pkm(event):
        x = event.x
        y = event.y
        Toe_R_mn.post(event.x_root, event.y_root)

    Toe_R_mn = Menu(tearoff=0)
    Toe_R_mn.add_command(label="3D X-position", command=lambda: show_graphic(Toe_R + " X", 'x'))
    Toe_R_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Toe_R + " Y", 'y'))
    Toe_R_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Toe_R + " Z", 'z'))
    Toe_R_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Toe_R, "x"))
    Toe_R_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Toe_R, "y"))
    Toe_R_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Toe_R, "z"))
    Toe_R_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Toe_R))


    Toe_R_lb = Label(frame_graphics, text=Toe_R, background="white", width=13, font=("Arial Bold", 14),
                     highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Toe_R_lb.place(relx=-0.001, rely=0.85)

    Toe_R_lb.bind('<Button-3>', Toe_R_pkm)


    # Toe_L_graphics -------------------------------------------------------------------------------

    def Toe_L_pkm(event):
        x = event.x
        y = event.y
        Toe_L_mn.post(event.x_root, event.y_root)

    Toe_L_mn = Menu(tearoff=0)
    Toe_L_mn.add_command(label="3D X-position", command=lambda: show_graphic(Toe_L + " X", 'x'))
    Toe_L_mn.add_command(label="3D Y-position", command=lambda: show_graphic(Toe_L + " Y", 'y'))
    Toe_L_mn.add_command(label="3D Z-position", command=lambda: show_graphic(Toe_L + " Z", 'z'))
    Toe_L_mn.add_command(label="X-axis speed", command=lambda: show_graphic_speed(Toe_L, "x"))
    Toe_L_mn.add_command(label="Y-axis speed", command=lambda: show_graphic_speed(Toe_L, "y"))
    Toe_L_mn.add_command(label="Z-axis speed", command=lambda: show_graphic_speed(Toe_L, "z"))
    Toe_L_mn.add_command(label="Absolute speed", command=lambda: show_graphic_speed(Toe_L))


    Toe_L_lb = Label(frame_graphics, text=Toe_L, background="white", width=13, font=("Arial Bold", 14),
                     highlightbackground="black", highlightcolor="black", highlightthickness=1)
    Toe_L_lb.place(relx=-0.001, rely=0.9)

    Toe_L_lb.bind('<Button-3>', Toe_L_pkm)

    first_point = ttk.Combobox(frame_graphics, values=Markers)
    second_point = ttk.Combobox(frame_graphics, values=Markers)
    third_point = ttk.Combobox(frame_graphics, values=Markers)
    extra_point = ttk.Combobox(frame_graphics, values=Markers + ["None"])
    extra_point.insert(0, "None")
    first_point.place(relx=0.4, rely=0.2)
    second_point.place(relx=0.4, rely=0.3)
    third_point.place(relx=0.4, rely=0.4)
    extra_point.place(relx=0.4, rely=0.5)


    def show_graphic_angle(first_point, second_point, third_point, extra_point):
        global window
        window = Tk()
        window.title("Выберите промежуток шагов")
        window.geometry("350x220+600+350")
        global steps_from_graf, steps_to_graf
        steps_from_graf = ttk.Combobox(window, width=8)
        steps_to_graf = ttk.Combobox(window, width=8)
        steps = list(range(1, m.num_steps))
        steps_from_graf["value"] = steps
        steps_from_cb.set('')
        steps_from_graf.insert(0, steps[0])
        steps_to_graf["value"] = steps
        steps_to_cb.set('')
        steps_to_graf.insert(0, steps[-1])
        superimposing_angle_lb = ttk.Label(window, text="Отображение графиков с наложением шагов")
        superimposing_angle_lb.place(relx=0.13, rely=0.44)

        superimposing_angle_cb = ttk.Combobox(window, values=["Да", "Нет"], width=8)
        superimposing_angle_cb.place(relx=0.4, rely=0.56)
        superimposing_angle_cb.set("Нет")


        lb_from = ttk.Label(window, text="C")
        lb_to = ttk.Label(window, text="ПО")
        lb_from.place(relx=0.28, rely=0.15)
        lb_to.place(relx=0.67, rely=0.15)

        steps_from_graf.place(relx=0.2, rely=0.27)
        steps_to_graf.place(relx=0.6, rely=0.27)

        steps_from_graf.bind("<<ComboboxSelected>>", steps_from_graf_selected_graphics)

        def return_graf():
            if superimposing_angle_cb.get() == "Да":
                return graphics_angle_100(first_point, second_point, third_point, extra_point, int(steps_from_graf.get()), int(steps_to_graf.get()))
            else:
                return graphics_angle(first_point, second_point, third_point, extra_point, int(steps_from_graf.get()), int(steps_to_graf.get()))

        save_steps_to_graf = ttk.Button(window, text="Вывести график", command=return_graf)
        save_steps_to_graf.place(relx=0.37, rely=0.8)

    def graphics_angle_100(first_point, second_point, third_point, extra_point, st_from, st_to):
        window.destroy()
        if extra_point == 'None':
            extra_point = None

        mas_x = []

        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]

        mas_y = [[] for i in range(st_from, st_to + 2)]

        for i in range(st_from, st_to + 1):
            mas_y[i - st_from] = m.angle_int(first_point, second_point, third_point, extra_point, fs + '_L' + str(i),
                                             fs + '_L' + str(i + 1))

        y = mas_y
        # print(x, "\n", y, "\n", len(x), len(y))
        plt.figure("Angle " + first_point + " - " + second_point + " - " + third_point + (
            " - " + extra_point if extra_point else ""))
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        plt.ylabel("Угол, в [град]")
        plt.grid()
        index = 0
        for i in y:
            x = np.linspace(0, 100, len(i))
            if index + st_from > st_to:
                continue
            plt.plot(x, i, label=f'{st_from + index} шаг')
            index += 1
        plt.legend()
        plt.show()

    def graphics_angle(first_point, second_point, third_point, extra_point, st_from, st_to):
        window.destroy()
        if extra_point == 'None':
            extra_point = None

        mas_x = []
        mas_y = []

        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]
            if i > st_from:
                mas_y.extend(m.angle_int(first_point, second_point, third_point, extra_point, fs + '_L' + str(i - 1), fs + '_L' + str(i)))

        x = np.linspace(0, (st_to - st_from + 1) * 100, mas_x[-1]-mas_x[0])
        y = mas_y
        #print(x, "\n", y, "\n", len(x), len(y))
        plt.figure("Angle " + first_point + " - " + second_point + " - " + third_point + (" - " + extra_point if extra_point else ""))
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        plt.ylabel("Угол, в [град]")
        plt.grid()
        plt.plot(x, y)
        plt.show()




    # построение графиков (скорость)

    def show_graphic_speed(data, axis=''):
        global window
        window = Tk()
        window.title("Выберите промежуток шагов")
        window.geometry("350x220+600+350")
        global steps_from_graf, steps_to_graf
        steps_from_graf = ttk.Combobox(window, width=8)
        steps_to_graf = ttk.Combobox(window, width=8)
        steps = list(range(1, m.num_steps))
        steps_from_graf["value"] = steps
        steps_from_cb.set('')
        steps_from_graf.insert(0, steps[0])
        steps_to_graf["value"] = steps
        steps_to_cb.set('')
        steps_to_graf.insert(0, steps[-1])

        superimposing_lb = ttk.Label(window, text="Отображение графиков с наложением шагов")
        superimposing_lb.place(relx=0.13, rely=0.44)

        superimposing_cb = ttk.Combobox(window, values=["Да", "Нет"], width=8)
        superimposing_cb.place(relx=0.4, rely=0.56)
        superimposing_cb.set("Нет")

        lb_from = ttk.Label(window, text="C")
        lb_to = ttk.Label(window, text="ПО")
        lb_from.place(relx=0.28, rely=0.15)
        lb_to.place(relx=0.67, rely=0.15)

        steps_from_graf.place(relx=0.2, rely=0.27)
        steps_to_graf.place(relx=0.6, rely=0.27)

        steps_from_graf.bind("<<ComboboxSelected>>", steps_from_graf_selected_graphics)

        def return_graf():
            if superimposing_cb.get() == "Да":
                return data_graphic_speed(int(steps_from_graf.get()), int(steps_to_graf.get()), data, axis)
            else:
                return data_graphic_speed_1(int(steps_from_graf.get()), int(steps_to_graf.get()), data, axis)

        save_steps_to_graf = ttk.Button(window, text="Вывести график", command=return_graf)
        save_steps_to_graf.place(relx=0.37, rely=0.8)

    def data_graphic_speed_1(st_from, st_to, data, axis):
        window.destroy()
        mas_x = []
        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]

        x = np.linspace(0, (st_to - st_from + 1) * 100, mas_x[-1] - mas_x[0])

        mas_y = m.st_vel_int(data, mas_x[0], mas_x[-1], axis)

        y = mas_y
        plt.figure(data + " " + axis + " speed  from " + str(st_from) + " to " + str(st_to) + " steps")
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        if axis != "":
            plt.ylabel(f"мм/с  (ось  {axis})")
        else:
            plt.ylabel("мм/с")
        plt.grid()
        plt.plot(x, y)
        plt.show()

    def data_graphic_speed(st_from, st_to, data, axis):
        window.destroy()
        mas_x = []
        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]

        mas_y = [[] for i in range(st_from, st_to + 2)]

        for k in range(st_from, st_to + 1):
            mas_y[k - st_from] = m.st_vel_int(data, mas_x[k - st_from], mas_x[k - st_from + 1], axis)

        #print(axis)
        #print(mas_y)

        y = mas_y
        plt.figure(data + " "+ axis + " speed  from " + str(st_from) + " to " + str(st_to) + " steps")
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        if axis != "":
            plt.ylabel(f"мм/с  (ось  {axis})")
        else:
            plt.ylabel("мм/с")
        plt.grid()
        index = 0
        for i in y:
            x = np.linspace(0, 100, len(i))
            if index + st_from > st_to:
                continue
            plt.plot(x, i, label=f'{st_from + index} шаг')
            index += 1
        plt.legend()
        plt.show()

    def show_graphic_speed_angle(first_point, second_point, third_point, extra_point):
        global window
        window = Tk()
        window.title("Выберите промежуток шагов")
        window.geometry("350x220+600+350")
        global steps_from_graf, steps_to_graf
        steps_from_graf = ttk.Combobox(window, width=8)
        steps_to_graf = ttk.Combobox(window, width=8)
        steps = list(range(1, m.num_steps))
        steps_from_graf["value"] = steps
        steps_from_cb.set('')
        steps_from_graf.insert(0, steps[0])
        steps_to_graf["value"] = steps
        steps_to_cb.set('')
        steps_to_graf.insert(0, steps[-1])
        superimposing_angle_lb = ttk.Label(window, text="Отображение графиков с наложением шагов")
        superimposing_angle_lb.place(relx=0.13, rely=0.44)

        superimposing_angle_cb = ttk.Combobox(window, values=["Да", "Нет"], width=8)
        superimposing_angle_cb.place(relx=0.4, rely=0.56)
        superimposing_angle_cb.set("Нет")

        lb_from = ttk.Label(window, text="C")
        lb_to = ttk.Label(window, text="ПО")
        lb_from.place(relx=0.28, rely=0.15)
        lb_to.place(relx=0.67, rely=0.15)

        steps_from_graf.place(relx=0.2, rely=0.27)
        steps_to_graf.place(relx=0.6, rely=0.27)

        steps_from_graf.bind("<<ComboboxSelected>>", steps_from_graf_selected_graphics)

        def return_graf():
            if superimposing_angle_cb.get() == "Да":
                return graphics_speed_angle_100(first_point, second_point, third_point, extra_point,
                                                int(steps_from_graf.get()),
                                                int(steps_to_graf.get()))
            else:
                return graphics_speed_angle(first_point, second_point, third_point, extra_point,
                                            int(steps_from_graf.get()),
                                            int(steps_to_graf.get()))

        save_steps_to_graf = ttk.Button(window, text="Вывести график", command=return_graf)
        save_steps_to_graf.place(relx=0.37, rely=0.8)

    def graphics_speed_angle_100(first_point, second_point, third_point, extra_point, st_from, st_to):
        window.destroy()
        if extra_point == 'None':
            extra_point = None

        mas_x = []

        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]

        mas_y = [[] for i in range(st_from, st_to + 2)]

        for i in range(st_from, st_to + 1):
            mas_y[i - st_from] = m.angle_int(first_point, second_point, third_point, extra_point, fs + '_L' + str(i),
                                             fs + '_L' + str(i + 1))
        print(mas_x[-1] - mas_x[0])
        y = mas_y
        # print(x, "\n", y, "\n", len(x), len(y))
        plt.figure("Angle speed " + first_point + " - " + second_point + " - " + third_point + (
            " - " + extra_point if extra_point else ""))
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        plt.ylabel("Скорость угла [рад/с]")
        plt.grid()
        index = 0
        for i in y:
            if index + st_from > st_to or len(i) == 0:
                continue
            x = np.linspace(0, 100, len(i)-1)
            y2 = m.angl_vel_int(i, mas_x[index], mas_x[index+1])
            plt.plot(x, y2, label=f'{st_from + index} шаг')
            index += 1
        plt.legend()
        plt.show()

    def graphics_speed_angle(first_point, second_point, third_point, extra_point, st_from, st_to):
        window.destroy()
        if extra_point == 'None':
            extra_point = None

        mas_x = []
        mas_y = []

        for i in range(st_from, st_to + 2):
            mas_x += [m.frame(fs + '_L' + str(i))]
            if i > st_from:
                mas_y.extend(m.angle_int(first_point, second_point, third_point, extra_point, fs + '_L' + str(i - 1),
                                         fs + '_L' + str(i)))

        x = np.linspace(0, (st_to - st_from + 1) * 100, mas_x[-1] - mas_x[0] - 1)
        y = m.angl_vel_int(mas_y, mas_x[0], mas_x[-1])
        # print(x, "\n", y, "\n", len(x), len(y))
        plt.figure("Angle " + first_point + " - " + second_point + " - " + third_point + (
            " - " + extra_point if extra_point else ""))
        plt.title(" ")
        plt.xlabel("Продолжительность шага, %")
        plt.ylabel("Скорость угла [рад/с]")
        plt.grid()
        plt.plot(x, y)
        plt.show()


    btn_graphics_angle = ttk.Button(frame_graphics, text="Построить график угла", command=lambda: show_graphic_angle(first_point.get(), second_point.get(), third_point.get(), extra_point.get()))
    btn_graphics_angle.place(relx=0.4, rely=0.6)
    btn_graphics_angle = ttk.Button(frame_graphics, text="Построить график скорости угла", command=lambda: show_graphic_speed_angle(first_point.get(), second_point.get(), third_point.get(), extra_point.get()))
    btn_graphics_angle.place(relx=0.4, rely=0.7)
    angle_lb = Label(frame_graphics, text='Выберите маркеры, по которым хотите построить угол', font=("Arial Bold", 14))
    angle_lb.place(relx=0.25, rely=0.1)
    marker1_lb = ttk.Label(frame_graphics, text='1 маркер')
    marker1_lb.place(relx=0.35, rely=0.2)
    marker2_lb = ttk.Label(frame_graphics, text='2 маркер')
    marker2_lb.place(relx=0.35, rely=0.3)
    marker3_lb = ttk.Label(frame_graphics, text='3 маркер')
    marker3_lb.place(relx=0.35, rely=0.4)
    marker4_lb = ttk.Label(frame_graphics, text='4 маркер')
    marker4_lb.place(relx=0.35, rely=0.5)

    btn_gr = ttk.Button(frame_graphics, text="Диаграмма шага", command=graf, style='my.TButton')
    btn_gr.place(relx=0.8, rely=0.43,width=200, height=50)

    global wb, ws

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Result')
    ws.write(0, 0, 'PID')
    ws.write(0, 1, 'Parameter')
    ws.write(0, 3, 'Value')
    ws.write(0, 5, 'Units')
    ws.write(0, 6, 'Alt. Value')
    ws.write(0, 8, 'Alt. Units')
    ws.write(1, 1, 'Данные')
    ws.write(2, 0, 1)
    ws.write(2, 1, 'Имя')
    ws.write(3, 0 , 2)
    ws.write(3, 1, 'Фамилия')
    ws.write(4, 0, 3)
    ws.write(4, 1, 'Скорость')
    ws.write(4, 3, speed)
    ws.write(4, 5, 'км/ч')
    ws.write(5, 0, 4)
    ws.write(5, 1, 'Рост')
    ws.write(5, 3, height)
    ws.write(5, 5, 'см')
    ws.write(6, 0, 5)
    ws.write(6, 1, 'Файл движения')
    ws.write(6, 3, model_path)
    ws.write(7, 0, 6)
    ws.write(7, 1, 'Файл статики')
    ws.write(7, 3, stat_path)
    ws.write(8, 0, 7)
    ws.write(8, 1, 'Версия программы')
    ws.write(8, 3, version)
    ws.write(10, 3, 'mean')
    ws.write(10, 4, 'std')
    ws.write(10, 6, 'mean')
    ws.write(10, 7, 'std')
    ws.write(11, 1, 'Параметры бегового шага')
    global line
    line = 11


    showinfo(title="Информация", message="Успешно", icon=INFO, default=OK)
    notebook.tab(frame_graphics, state=NORMAL)
    notebook.tab(frame_export, state=NORMAL)
    notebook.select(frame_graphics)






from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import OK, INFO, showinfo

def static_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File")
    path_stat_tf.delete(0, END)
    path_stat_tf.insert(0, filename)


path = 0
m = 0


def din_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File")
    path_din_tf.delete(0, END)
    path_din_tf.insert(0, filename)
    if path_din_tf.get():
        global path, m
        path = path_din_tf.get()
        m = Model()
        m.load_model(path, key_moments)
        m_events = m.__events__.name.to_list()
        if m_events[-1][-1] not in '0123456789':
            m.numerate_events()
        m.strip_data()
        steps = list(range(1, m.num_steps + 1))
        steps_from_cb["state"] = NORMAL
        steps_to_cb["state"] = NORMAL
        steps_from_cb["value"] = steps
        steps_from_cb.set('')
        steps_from_cb.insert(0, steps[0])
        steps_to_cb["value"] = steps
        steps_to_cb.set('')
        steps_to_cb.insert(0, steps[-1])


def steps_from_selected(*args):
    if (steps_to_cb.get() < steps_from_cb.get()):
        steps_to_cb["textvariable"] = steps_from_cb.get()
    steps_to_cb["value"] = list(range(int(steps_from_cb.get()), m.num_steps))


def save_xls_file():
    filename = filedialog.asksaveasfilename(initialdir="/", title="Select a File")
    export_save_tf.delete(0, END)
    export_save_tf.insert(0, filename)


root = Tk()
root.title("Биомеханический Анализ Локомоций")
root.geometry("1400x650+80+80")


s = ttk.Style()
s.configure('my.TButton', font=("Arial Bold", 11))
s.configure('my.TRadiobutton', font=("Arial Bold", 13))
s.configure('my.TCheckbutton', font=("Arial Bold", 11))

from picture import setting, picture, nast


photo = PhotoImage(data=setting)

root.iconbitmap(default=setting)


notebook = ttk.Notebook(width=1400, height=650)
notebook.pack(expand=True)

frame = Frame(
    notebook,
    padx=390,
    pady=185,
)
frame.pack()

photo2 = PhotoImage(data=picture)

photo3 = PhotoImage(data=nast)

picture_lb3 = ttk.Label(notebook, image=photo3)
picture_lb3.place(relx=0.05, rely=0.2)

type_move_lb = ttk.Label(notebook, text="Выберите тип движения:", font=("Arial Bold", 13))

type_move = StringVar(value="Бег")

run_rb = ttk.Radiobutton(notebook, text="Бег", value="Бег", variable=type_move, state=ACTIVE, style="my.TRadiobutton")
walk_rb = ttk.Radiobutton(notebook, text="Ходьба", value="Ходьба", variable=type_move, style="my.TRadiobutton")

type_move_lb.place(relx=0.75, rely=0.85)
run_rb.place(relx=0.75, rely=0.91)
walk_rb.place(relx=0.84, rely=0.91)

path_stat_lb = Label(
    frame,
    pady=10,
    text="Файл статики .tsv",
    font=("Arial Bold", 13)
)
path_stat_lb.grid(row=3, column=1)

check_stat = IntVar(value=0)
flag_stat_btn = ttk.Checkbutton(notebook, text="Не использовать статический файл", variable=check_stat, style="my.TCheckbutton")
flag_stat_btn.place(relx=0.78, rely=0.405)

path_din_lb = Label(
    frame,
    pady=10,
    text="Файл движения .tsv",
    font=("Arial Bold", 13)
)
path_din_lb.grid(row=2, column=1)

speed_lb = Label(
    frame,
    pady=10,
    text="Cкорость",
    font=("Arial Bold", 13)
)
speed_lb.grid(row=4, column=1)

height_lb = Label(
    frame,
    pady=10,
    text="Рост",
    font=("Arial Bold", 13)
)
height2_lb = Label(frame, text="см", font=("Arial Bold", 13))
height2_lb.place(relx=0.5, rely=0.54)
height_lb.grid(row=5, column=1)

steps_lb = Label(
    frame,
    pady=10,
    text="Диапазон  шагов  с                 шага    по                 шаг",
    font=("Arial Bold", 13)
)
steps_lb.place(relx=0.05, rely=0.7)

# path_save_lb = Label(
#     frame,
#     pady=10,
#     text="Сохранить в .xls",
# )
# path_save_lb.grid(row=5, column=1)

path_stat_tf = ttk.Entry(
    frame,
    width=60
)
path_stat_tf.grid(row=3, column=2)


path_din_tf = ttk.Entry(
    frame,
    width=60
)
path_din_tf.grid(row=2, column=2, pady=5, padx=20)

speed_tf = ttk.Spinbox(
    frame,
    from_=0.0,
    to=40.0,
    width=15,
)
speed_tf.grid(row=4, column=2, pady=5, sticky=W, padx=20)

miles = "миль/ч"
km = "км/ч"
mvs = "м/с"
speed_flag = StringVar(value=km)

miles_btn = ttk.Radiobutton(frame, text=miles, value=miles, variable=speed_flag, style="my.TRadiobutton")
miles_btn.place(relx=.765, rely=.43, anchor="c")

km_btn = ttk.Radiobutton(frame, text=km, value=km, variable=speed_flag, style="my.TRadiobutton")
km_btn.place(relx=.55, rely=.43, anchor="c")

km_btn = ttk.Radiobutton(frame, text=mvs, value=mvs, variable=speed_flag, style="my.TRadiobutton")
km_btn.place(relx=.65, rely=.43, anchor="c")


height_tf = ttk.Spinbox(
    frame,
    from_=0.0,
    to=220.0,
    width=15,
)
height_tf.grid(row=5, column=2, pady=5, sticky=W, padx=20)

# path_save_tf = ttk.Entry(
#     frame,
#     width=60
# )
# path_save_tf.grid(row=5, column=2, pady=5)

btn_for_static_file = ttk.Button(
    frame,
    text='Обзор',
    command=static_file
)
btn_for_static_file.grid(row=3, column=3)

btn_for_din_file = ttk.Button(
    frame,
    text='Обзор',
    command=din_file
)
btn_for_din_file.grid(row=2, column=3)

steps_from_cb = ttk.Combobox(frame, state=DISABLED, width=8)
steps_to_cb = ttk.Combobox(frame, state=DISABLED, width=8)

steps_from_cb.place(relx=0.32, rely=0.75)
steps_to_cb.place(relx=0.59, rely=0.75)

steps_from_cb.bind("<<ComboboxSelected>>", steps_from_selected)
# btn_for_save_xls_file = ttk.Button(
#     frame,
#     text='Обзор',
#     command=save_xls_file
# )
# btn_for_save_xls_file.grid(row=5, column=3)

cal_btn = ttk.Button(
    frame,
    text='Рассчитать параметры',
    command=main,
    style='my.TButton'
)
cal_btn.place(relx=0.35, rely=0.95, width=200, height=50)



frame_parameters = Frame(
    notebook
)
frame_parameters.pack(expand=True)

canvas = Canvas(frame_parameters, width=1400, height=650)
canvas.pack(anchor=CENTER, expand=1)
canvas.create_rectangle(20, 20, 300, 195)
canvas.create_rectangle(320, 20, 600, 170)
canvas.create_rectangle(620, 20, 900, 170)
canvas.create_rectangle(920, 20, 1200, 120)
canvas.create_rectangle(20, 225, 422, 325)
canvas.create_rectangle(442, 225, 855, 325)
canvas.create_rectangle(875, 225, 1330, 325)
canvas.create_rectangle(430, 366, 900, 516)

def running_step_parameter():
    if check_running_step_parameter.get() == 1:
        check_parameter_1.set(value=1)
        check_parameter_2.set(value=1)
        check_parameter_3.set(value=1)
        check_parameter_4.set(value=1)
        check_parameter_5.set(value=1)
        check_parameter_6.set(value=1)
        parameter_1_btn["state"] = ACTIVE
        parameter_2_btn["state"] = ACTIVE
        parameter_3_btn["state"] = ACTIVE
        parameter_4_btn["state"] = ACTIVE
        parameter_5_btn["state"] = ACTIVE
        parameter_6_btn["state"] = ACTIVE
    else:
        check_parameter_1.set(value=0)
        check_parameter_2.set(value=0)
        check_parameter_3.set(value=0)
        check_parameter_4.set(value=0)
        check_parameter_5.set(value=0)
        check_parameter_6.set(value=0)
        parameter_1_btn["state"] = DISABLED
        parameter_2_btn["state"] = DISABLED
        parameter_3_btn["state"] = DISABLED
        parameter_4_btn["state"] = DISABLED
        parameter_5_btn["state"] = DISABLED
        parameter_6_btn["state"] = DISABLED


check_running_step_parameter = IntVar(value=1)
running_step_parameter_btn = ttk.Checkbutton(frame_parameters, text="Параметры бегового шага:", variable=check_running_step_parameter, command=running_step_parameter)
running_step_parameter_btn.place(relx=0.05, rely=0.02)

# Проверка на выбор параметра:  Частота шагов
check_parameter_1 = IntVar(value=1)
parameter_1_btn = ttk.Checkbutton(frame_parameters, text="Частота шагов", variable=check_parameter_1)
parameter_1_btn.place(relx=0.03, rely=0.06)

# Проверка на выбор параметра:  Длина шага
check_parameter_2 = IntVar(value=1)
parameter_2_btn = ttk.Checkbutton(frame_parameters, text="Длина шага", variable=check_parameter_2)
parameter_2_btn.place(relx=0.03, rely=0.1)

# Проверка на выбор параметра:  Длительность фазы опоры
check_parameter_3 = IntVar(value=1)
parameter_3_btn = ttk.Checkbutton(frame_parameters, text="Длительность фазы опоры", variable=check_parameter_3)
parameter_3_btn.place(relx=0.03, rely=0.14)

# Проверка на выбор параметра:  Длительность фазы полета
check_parameter_4 = IntVar(value=1)
parameter_4_btn = ttk.Checkbutton(frame_parameters, text="Длительность фазы полета", variable=check_parameter_4)
parameter_4_btn.place(relx=0.03, rely=0.18)

# Проверка на выбор параметра:  Длительность фазы аммортизации
check_parameter_5 = IntVar(value=1)
parameter_5_btn = ttk.Checkbutton(frame_parameters, text="Длительность фазы аммортизации", variable=check_parameter_5)
parameter_5_btn.place(relx=0.03, rely=0.22)

# Проверка на выбор параметра:  Длительность фазы отталкивания
check_parameter_6 = IntVar(value=1)
parameter_6_btn = ttk.Checkbutton(frame_parameters, text="Длительность фазы отталкивания", variable=check_parameter_6)
parameter_6_btn.place(relx=0.03, rely=0.26)


def body_movement():
    if check_body_movement.get() == 1:
        check_parameter_7.set(value=1)
        check_parameter_8.set(value=1)
        check_parameter_9.set(value=1)
        check_parameter_10.set(value=1)
        check_parameter_11.set(value=1)
        parameter_7_btn["state"] = ACTIVE
        parameter_8_btn["state"] = ACTIVE
        parameter_9_btn["state"] = ACTIVE
        parameter_10_btn["state"] = ACTIVE
        parameter_11_btn["state"] = ACTIVE
    else:
        check_parameter_7.set(value=0)
        check_parameter_8.set(value=0)
        check_parameter_9.set(value=0)
        check_parameter_10.set(value=0)
        check_parameter_11.set(value=0)
        parameter_7_btn["state"] = DISABLED
        parameter_8_btn["state"] = DISABLED
        parameter_9_btn["state"] = DISABLED
        parameter_10_btn["state"] = DISABLED
        parameter_11_btn["state"] = DISABLED

check_body_movement = IntVar(value=1)
body_movement_btn = ttk.Checkbutton(frame_parameters, text="Движение корпуса:", variable=check_body_movement, command=body_movement)
body_movement_btn.place(relx=0.28, rely=0.02)

# Проверка на выбор параметра:  Размах вертикальных колебаний
check_parameter_7 = IntVar(value=1)
parameter_7_btn = ttk.Checkbutton(frame_parameters, text="Размах вертикальных колебаний", variable=check_parameter_7)
parameter_7_btn.place(relx=0.24, rely=0.06)

# Проверка на выбор параметра:  Угол наклона корпуса вперед-назад
check_parameter_8 = IntVar(value=1)
parameter_8_btn = ttk.Checkbutton(frame_parameters, text="Угол наклона корпуса вперед-назад", variable=check_parameter_8)
parameter_8_btn.place(relx=0.24, rely=0.1)

# Проверка на выбор параметра:  Угол наклона корпуса вправо-влево
check_parameter_9 = IntVar(value=1)
parameter_9_btn = ttk.Checkbutton(frame_parameters, text="Угол наклона корпуса вправо-влево", variable=check_parameter_9)
parameter_9_btn.place(relx=0.24, rely=0.14)

# Проверка на выбор параметра:  Угол вращения таза
check_parameter_10 = IntVar(value=1)
parameter_10_btn = ttk.Checkbutton(frame_parameters, text="Угол вращения таза", variable=check_parameter_10)
parameter_10_btn.place(relx=0.24, rely=0.18)

# Проверка на выбор параметра:  Угол вращения плечевого пояса
check_parameter_11 = IntVar(value=1)
parameter_11_btn = ttk.Checkbutton(frame_parameters, text="Угол вращения плечевого пояса", variable=check_parameter_11)
parameter_11_btn.place(relx=0.24, rely=0.22)


def hand_movement():
    if check_hand_movement.get() == 1:
        check_parameter_12.set(value=1)
        check_parameter_13.set(value=1)
        check_parameter_14.set(value=1)
        check_parameter_15.set(value=1)
        check_parameter_16.set(value=1)
        parameter_12_btn["state"] = ACTIVE
        parameter_13_btn["state"] = ACTIVE
        parameter_14_btn["state"] = ACTIVE
        parameter_15_btn["state"] = ACTIVE
        parameter_16_btn["state"] = ACTIVE
    else:
        check_parameter_12.set(value=0)
        check_parameter_13.set(value=0)
        check_parameter_14.set(value=0)
        check_parameter_15.set(value=0)
        check_parameter_16.set(value=0)
        parameter_12_btn["state"] = DISABLED
        parameter_13_btn["state"] = DISABLED
        parameter_14_btn["state"] = DISABLED
        parameter_15_btn["state"] = DISABLED
        parameter_16_btn["state"] = DISABLED

check_hand_movement = IntVar(value=1)
hand_movement_btn = ttk.Checkbutton(frame_parameters, text="Движение рук:", variable=check_hand_movement, command=hand_movement)
hand_movement_btn.place(relx=0.51, rely=0.02)

# Проверка на выбор параметра:  Угол в локтевом суставе
check_parameter_12 = IntVar(value=1)
parameter_12_btn = ttk.Checkbutton(frame_parameters, text="Угол в локтевом суставе", variable=check_parameter_12)
parameter_12_btn.place(relx=0.45, rely=0.06)

# Проверка на выбор параметра:  Угол сгибания плеча
check_parameter_13 = IntVar(value=1)
parameter_13_btn = ttk.Checkbutton(frame_parameters, text="Угол сгибания плеча", variable=check_parameter_13)
parameter_13_btn.place(relx=0.45, rely=0.1)

# Проверка на выбор параметра:  Угол разгибания плеча
check_parameter_14 = IntVar(value=1)
parameter_14_btn = ttk.Checkbutton(frame_parameters, text="Угол разгибания плеча", variable=check_parameter_14)
parameter_14_btn.place(relx=0.45, rely=0.14)

# Проверка на выбор параметра:  Угол отведения плеча
check_parameter_15 = IntVar(value=1)
parameter_15_btn = ttk.Checkbutton(frame_parameters, text="Угол отведения плеча", variable=check_parameter_15)
parameter_15_btn.place(relx=0.45, rely=0.18)

# Проверка на выбор параметра:  Длина траектории кисти
check_parameter_16 = IntVar(value=1)
parameter_16_btn = ttk.Checkbutton(frame_parameters, text="Длина траектории кисти", variable=check_parameter_16)
parameter_16_btn.place(relx=0.45, rely=0.22)


def flywheel_leg():
    if check_flywheel_leg.get() == 1:
        check_parameter_17.set(value=1)
        check_parameter_18.set(value=1)
        check_parameter_19.set(value=1)
        parameter_17_btn["state"] = ACTIVE
        parameter_18_btn["state"] = ACTIVE
        parameter_19_btn["state"] = ACTIVE
    else:
        check_parameter_17.set(value=0)
        check_parameter_18.set(value=0)
        check_parameter_19.set(value=0)
        parameter_17_btn["state"] = DISABLED
        parameter_18_btn["state"] = DISABLED
        parameter_19_btn["state"] = DISABLED


check_flywheel_leg = IntVar(value=1)
flywheel_leg_btn = ttk.Checkbutton(frame_parameters, text="Движение маховой ноги:", variable=check_flywheel_leg, command=flywheel_leg)
flywheel_leg_btn.place(relx=0.70, rely=0.02)

# Проверка на выбор параметра:  Угол складывания голени
check_parameter_17 = IntVar(value=1)
parameter_17_btn = ttk.Checkbutton(frame_parameters, text="Угол складывания голени", variable=check_parameter_17)
parameter_17_btn.place(relx=0.67, rely=0.06)

# Проверка на выбор параметра:  Угол выноса бедра
check_parameter_18 = IntVar(value=1)
parameter_18_btn = ttk.Checkbutton(frame_parameters, text="Угол выноса бедра", variable=check_parameter_18)
parameter_18_btn.place(relx=0.67, rely=0.1)

# Проверка на выбор параметра:  Угол сведения бедер и фазовый путь бедра
check_parameter_19 = IntVar(value=1)
parameter_19_btn = ttk.Checkbutton(frame_parameters, text="Угол сведения бедер и фазовый путь бедра", variable=check_parameter_19)
parameter_19_btn.place(relx=0.67, rely=0.14)


def support_leg():
    if check_support_leg.get() == 1:
        check_parameter_20.set(value=1)
        check_parameter_21.set(value=1)
        check_parameter_22.set(value=1)
        parameter_20_btn["state"] = ACTIVE
        parameter_21_btn["state"] = ACTIVE
        parameter_22_btn["state"] = ACTIVE
    else:
        check_parameter_20.set(value=0)
        check_parameter_21.set(value=0)
        check_parameter_22.set(value=0)
        parameter_20_btn["state"] = DISABLED
        parameter_21_btn["state"] = DISABLED
        parameter_22_btn["state"] = DISABLED


check_support_leg = IntVar(value=1)
support_leg_btn = ttk.Checkbutton(frame_parameters, text="Движение опорной ноги при постановке на опору:", variable=check_support_leg, command=support_leg)
support_leg_btn.place(relx=0.05, rely=0.35)

# Проверка на выбор параметра:  Угол в коленном суставе
check_parameter_20 = IntVar(value=1)
parameter_20_btn = ttk.Checkbutton(frame_parameters, text="Угол в коленном суставе", variable=check_parameter_20)
parameter_20_btn.place(relx=0.03, rely=0.39)

# Проверка на выбор параметра:  Угол в тазобедренном суставе
check_parameter_21 = IntVar(value=1)
parameter_21_btn = ttk.Checkbutton(frame_parameters, text="Угол в тазобедренном суставе", variable=check_parameter_21)
parameter_21_btn.place(relx=0.03, rely=0.43)

# Проверка на выбор параметра:  Вынос стопы
check_parameter_22 = IntVar(value=1)
parameter_22_btn = ttk.Checkbutton(frame_parameters, text="Вынос стопы", variable=check_parameter_22)
parameter_22_btn.place(relx=0.03, rely=0.47)


def vertical_support_leg():
    if check_vertical_support_leg.get() == 1:
        check_parameter_23.set(value=1)
        check_parameter_24.set(value=1)
        check_parameter_25.set(value=1)
        parameter_23_btn["state"] = ACTIVE
        parameter_24_btn["state"] = ACTIVE
        parameter_25_btn["state"] = ACTIVE
    else:
        check_parameter_23.set(value=0)
        check_parameter_24.set(value=0)
        check_parameter_25.set(value=0)
        parameter_23_btn["state"] = DISABLED
        parameter_24_btn["state"] = DISABLED
        parameter_25_btn["state"] = DISABLED


check_vertical_support_leg = IntVar(value=1)
vertical_support_leg_btn = ttk.Checkbutton(frame_parameters, text="Движение опорной ноги при прохождении вертикали:", variable=check_vertical_support_leg, command=vertical_support_leg)
vertical_support_leg_btn.place(relx=0.35, rely=0.35)

# Проверка на выбор параметра:  Угол в коленном суставе
check_parameter_23 = IntVar(value=1)
parameter_23_btn = ttk.Checkbutton(frame_parameters, text="Угол в коленном суставе", variable=check_parameter_23)
parameter_23_btn.place(relx=0.33, rely=0.39)

# Проверка на выбор параметра:  Угол в тазобедренном суставе
check_parameter_24 = IntVar(value=1)
parameter_24_btn = ttk.Checkbutton(frame_parameters, text="Угол в тазобедренном суставе", variable=check_parameter_24)
parameter_24_btn.place(relx=0.33, rely=0.43)

# Проверка на выбор параметра:  Высота пятки
check_parameter_25 = IntVar(value=1)
parameter_25_btn = ttk.Checkbutton(frame_parameters, text="Высота пятки", variable=check_parameter_25)
parameter_25_btn.place(relx=0.33, rely=0.47)


def support_leg_repulsion():
    if check_support_leg_repulsion.get() == 1:
        check_parameter_26.set(value=1)
        check_parameter_27.set(value=1)
        check_parameter_28.set(value=1)
        parameter_26_btn["state"] = ACTIVE
        parameter_27_btn["state"] = ACTIVE
        parameter_28_btn["state"] = ACTIVE
    else:
        check_parameter_26.set(value=0)
        check_parameter_27.set(value=0)
        check_parameter_28.set(value=0)
        parameter_26_btn["state"] = DISABLED
        parameter_27_btn["state"] = DISABLED
        parameter_28_btn["state"] = DISABLED


check_support_leg_repulsion = IntVar(value=1)
support_leg_repulsion_btn = ttk.Checkbutton(frame_parameters, text="Движение опорной ноги при завершении отталкивания:", variable=check_support_leg_repulsion, command=support_leg_repulsion)
support_leg_repulsion_btn.place(relx=0.67, rely=0.35)

# Проверка на выбор параметра:  Угол в коленном суставе
check_parameter_26 = IntVar(value=1)
parameter_26_btn = ttk.Checkbutton(frame_parameters, text="Угол в коленном суставе", variable=check_parameter_26)
parameter_26_btn.place(relx=0.64, rely=0.39)

# Проверка на выбор параметра:  Угол в голеностопном суставе
check_parameter_27 = IntVar(value=1)
parameter_27_btn = ttk.Checkbutton(frame_parameters, text="Угол в голеностопном суставе", variable=check_parameter_27)
parameter_27_btn.place(relx=0.64, rely=0.43)

# Проверка на выбор параметра:  Угол постановки ноги (между голенью и опорой)
check_parameter_28 = IntVar(value=1)
parameter_28_btn = ttk.Checkbutton(frame_parameters, text="Угол постановки ноги (между голенью и опорой)", variable=check_parameter_28)
parameter_28_btn.place(relx=0.64, rely=0.47)


def speed_indicators():
    if check_speed_indicators.get() == 1:
        check_parameter_29.set(value=1)
        check_parameter_30.set(value=1)
        check_parameter_31.set(value=1)
        check_parameter_32.set(value=1)
        check_parameter_33.set(value=1)
        parameter_29_btn["state"] = ACTIVE
        parameter_30_btn["state"] = ACTIVE
        parameter_31_btn["state"] = ACTIVE
        parameter_32_btn["state"] = ACTIVE
        parameter_33_btn["state"] = ACTIVE
    else:
        check_parameter_29.set(value=0)
        check_parameter_30.set(value=0)
        check_parameter_31.set(value=0)
        check_parameter_32.set(value=0)
        check_parameter_33.set(value=0)
        parameter_29_btn["state"] = DISABLED
        parameter_30_btn["state"] = DISABLED
        parameter_31_btn["state"] = DISABLED
        parameter_32_btn["state"] = DISABLED
        parameter_33_btn["state"] = DISABLED


check_speed_indicators = IntVar(value=1)
speed_indicators_btn = ttk.Checkbutton(frame_parameters, text="Показатели скорости:", variable=check_speed_indicators, command=speed_indicators)
speed_indicators_btn.place(relx=0.43, rely=0.57)

# Проверка на выбор параметра:  Посадочная скорость стопы
check_parameter_29 = IntVar(value=1)
parameter_29_btn = ttk.Checkbutton(frame_parameters, text="Посадочная скорость стопы", variable=check_parameter_29)
parameter_29_btn.place(relx=0.32, rely=0.61)

# Проверка на выбор параметра:  Угловая скорость голени при постановке стопы на опору
check_parameter_30 = IntVar(value=1)
parameter_30_btn = ttk.Checkbutton(frame_parameters, text="Угловая скорость голени при постановке стопы на опору", variable=check_parameter_30)
parameter_30_btn.place(relx=0.32, rely=0.65)

# Проверка на выбор параметра:  Угловая скорость бедра при постановке стопы на опору
check_parameter_31 = IntVar(value=1)
parameter_31_btn = ttk.Checkbutton(frame_parameters, text="Угловая скорость бедра при постановке стопы на опору", variable=check_parameter_31)
parameter_31_btn.place(relx=0.32, rely=0.69)

# Проверка на выбор параметра:  Изменение скорости тела
check_parameter_32 = IntVar(value=1)
parameter_32_btn = ttk.Checkbutton(frame_parameters, text="Изменение скорости тела", variable=check_parameter_32)
parameter_32_btn.place(relx=0.32, rely=0.73)

# Проверка на выбор параметра:  Потеря скорости тела в фазе амортизации
check_parameter_33 = IntVar(value=1)
parameter_33_btn = ttk.Checkbutton(frame_parameters, text="Потеря скорости тела в фазе амортизации", variable=check_parameter_33)
parameter_33_btn.place(relx=0.32, rely=0.77)


def on_all():
    if not check_parameter_1.get(): parameter_1_btn.invoke()
    if not check_parameter_2.get(): parameter_2_btn.invoke()
    if not check_parameter_3.get(): parameter_3_btn.invoke()
    if not check_parameter_4.get(): parameter_4_btn.invoke()
    if not check_parameter_5.get(): parameter_5_btn.invoke()
    if not check_parameter_6.get(): parameter_6_btn.invoke()
    if not check_parameter_7.get(): parameter_7_btn.invoke()
    if not check_parameter_8.get(): parameter_8_btn.invoke()
    if not check_parameter_9.get(): parameter_9_btn.invoke()
    if not check_parameter_10.get(): parameter_10_btn.invoke()
    if not check_parameter_11.get(): parameter_11_btn.invoke()
    if not check_parameter_12.get(): parameter_12_btn.invoke()
    if not check_parameter_13.get(): parameter_13_btn.invoke()
    if not check_parameter_14.get(): parameter_14_btn.invoke()
    if not check_parameter_15.get(): parameter_15_btn.invoke()
    if not check_parameter_16.get(): parameter_16_btn.invoke()
    if not check_parameter_17.get(): parameter_17_btn.invoke()
    if not check_parameter_18.get(): parameter_18_btn.invoke()
    if not check_parameter_19.get(): parameter_19_btn.invoke()
    if not check_parameter_20.get(): parameter_20_btn.invoke()
    if not check_parameter_21.get(): parameter_21_btn.invoke()
    if not check_parameter_22.get(): parameter_22_btn.invoke()
    if not check_parameter_23.get(): parameter_23_btn.invoke()
    if not check_parameter_24.get(): parameter_24_btn.invoke()
    if not check_parameter_25.get(): parameter_25_btn.invoke()
    if not check_parameter_26.get(): parameter_26_btn.invoke()
    if not check_parameter_27.get(): parameter_27_btn.invoke()
    if not check_parameter_28.get(): parameter_28_btn.invoke()
    if not check_parameter_29.get(): parameter_29_btn.invoke()
    if not check_parameter_30.get(): parameter_30_btn.invoke()
    if not check_parameter_31.get(): parameter_31_btn.invoke()
    if not check_parameter_32.get(): parameter_32_btn.invoke()
    if not check_parameter_33.get(): parameter_33_btn.invoke()
    on_all_value()


def on_all_value():
    check_parameter_1, check_parameter_2, check_parameter_3, check_parameter_4, check_parameter_5 = IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1)
    check_parameter_6, check_parameter_7, check_parameter_8, check_parameter_9, check_parameter_10 = IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1)
    check_parameter_11, check_parameter_12, check_parameter_13, check_parameter_15, check_parameter_16 = IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1)
    check_parameter_17, check_parameter_18, check_parameter_19, check_parameter_20, check_parameter_21 = IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1)
    check_parameter_22, check_parameter_23, check_parameter_24, check_parameter_25, check_parameter_26 = IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1)
    check_parameter_27, check_parameter_28, check_parameter_29, check_parameter_30, check_parameter_31, check_parameter_32 = IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1), IntVar(value=1)
    check_parameter_33 = IntVar(value=1)


def off_all():
    if check_parameter_1.get(): parameter_1_btn.invoke()
    if check_parameter_2.get(): parameter_2_btn.invoke()
    if check_parameter_3.get(): parameter_3_btn.invoke()
    if check_parameter_4.get(): parameter_4_btn.invoke()
    if check_parameter_5.get(): parameter_5_btn.invoke()
    if check_parameter_6.get(): parameter_6_btn.invoke()
    if check_parameter_7.get(): parameter_7_btn.invoke()
    if check_parameter_8.get(): parameter_8_btn.invoke()
    if check_parameter_9.get(): parameter_9_btn.invoke()
    if check_parameter_10.get(): parameter_10_btn.invoke()
    if check_parameter_11.get(): parameter_11_btn.invoke()
    if check_parameter_12.get(): parameter_12_btn.invoke()
    if check_parameter_13.get(): parameter_13_btn.invoke()
    if check_parameter_14.get(): parameter_14_btn.invoke()
    if check_parameter_15.get(): parameter_15_btn.invoke()
    if check_parameter_16.get(): parameter_16_btn.invoke()
    if check_parameter_17.get(): parameter_17_btn.invoke()
    if check_parameter_18.get(): parameter_18_btn.invoke()
    if check_parameter_19.get(): parameter_19_btn.invoke()
    if check_parameter_20.get(): parameter_20_btn.invoke()
    if check_parameter_21.get(): parameter_21_btn.invoke()
    if check_parameter_22.get(): parameter_22_btn.invoke()
    if check_parameter_23.get(): parameter_23_btn.invoke()
    if check_parameter_24.get(): parameter_24_btn.invoke()
    if check_parameter_25.get(): parameter_25_btn.invoke()
    if check_parameter_26.get(): parameter_26_btn.invoke()
    if check_parameter_27.get(): parameter_27_btn.invoke()
    if check_parameter_28.get(): parameter_28_btn.invoke()
    if check_parameter_29.get(): parameter_29_btn.invoke()
    if check_parameter_30.get(): parameter_30_btn.invoke()
    if check_parameter_31.get(): parameter_31_btn.invoke()
    if check_parameter_32.get(): parameter_32_btn.invoke()
    if check_parameter_33.get(): parameter_33_btn.invoke()
    off_all_value()


def off_all_value():
    check_parameter_1, check_parameter_2, check_parameter_3, check_parameter_4, check_parameter_5 = IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0)
    check_parameter_6, check_parameter_7, check_parameter_8, check_parameter_9, check_parameter_10 = IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0)
    check_parameter_11, check_parameter_12, check_parameter_13, check_parameter_15, check_parameter_16 = IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0)
    check_parameter_17, check_parameter_18, check_parameter_19, check_parameter_20, check_parameter_21 = IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0)
    check_parameter_22, check_parameter_23, check_parameter_24, check_parameter_25, check_parameter_26 = IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0)
    check_parameter_27, check_parameter_28, check_parameter_29, check_parameter_30, check_parameter_31, check_parameter_32 = IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0), IntVar(value=0)
    check_parameter_33 = IntVar(value=0)


btn_off = ttk.Button(frame_parameters, text="Выключить всё", command=off_all)
btn_on = ttk.Button(frame_parameters, text="Включить всё", command=on_all)
btn_off.place(relx=0.4, rely=0.95)
btn_on.place(relx=0.5, rely=0.95)


# Ключевые моменты
fs_var = StringVar(value="fs")
ms_var = StringVar(value="ms")
to_var = StringVar(value="to")

fs = fs_var.get()
ms = ms_var.get()
to = to_var.get()

# Маркеры
Back_T_var, Back_B_var = StringVar(value='Back_T'), StringVar(value='Back_B')  # верхняя и нижняя части спины
Shoulder_R_var, Shoulder_L_var = StringVar(value='Shoulder_R'), StringVar(value='Shoulder_L')  # плечевые суставы
Elbow_R_var, Elbow_L_var = StringVar(value='Elbow_R'), StringVar(value='Elbow_L')  # локтевые суставы
Wrist_R_var, Wrist_L_var = StringVar(value='Wrist_R'), StringVar(value='Wrist_L')  # лучезапястные суставы
Hip_R_var, Hip_L_var = StringVar(value='Hip_R'), StringVar(value='Hip_L')  # тазобедренные суставы
Knee_R_var, Knee_L_var = StringVar(value='Knee_R'), StringVar(value='Knee_L')  # коленные суставы
Ankle_R_var, Ankle_L_var = StringVar(value='Ankle_R'), StringVar(value='Ankle_L')  # голеностопные суставы
Heel_R_var, Heel_L_var = StringVar(value='Heel_R'), StringVar(value='Heel_L')  # пятки
Toe_R_var, Toe_L_var = StringVar(value='Toe_R'), StringVar(value='Toe_L')  # носки

Back_T, Back_B = Back_T_var.get(), Back_B_var.get()
Shoulder_R, Shoulder_L = Shoulder_R_var.get(), Shoulder_L_var.get()
Elbow_R, Elbow_L = Elbow_R_var.get(), Elbow_L_var.get()
Wrist_R, Wrist_L = Wrist_R_var.get(), Wrist_L_var.get()
Hip_R, Hip_L = Hip_R_var.get(), Hip_L_var.get()
Knee_R, Knee_L = Knee_R_var.get(), Knee_L_var.get()
Ankle_R, Ankle_L = Ankle_R_var.get(), Ankle_L_var.get()
Heel_R, Heel_L = Heel_R_var.get(), Heel_L_var.get()
Toe_R, Toe_L = Toe_R_var.get(), Toe_L_var.get()

# Направление
direction_var = StringVar(value="-x")


#Стандартные значения (ключевые моменты, маркеры и напровление)
key_moments = [fs, ms, to]
Markers = [Back_T, Back_B, Shoulder_R, Shoulder_L, Elbow_R, Elbow_L, Wrist_R,
           Wrist_L, Hip_R, Hip_L, Knee_R, Knee_L, Ankle_R, Ankle_L, Heel_R,
           Heel_L, Toe_R, Toe_L]
direction = direction_var.get()

unnumerate_var = StringVar(value="Да")

unnumerate_events_flag = unnumerate_var.get()

def save_data(window):
    # Сохранение данных (ключевые моменты, маркеры и напровление)
    global key_moments, fs, ms, to
    fs = fs_var.get()
    ms = ms_var.get()
    to = to_var.get()
    key_moments = [fs, ms, to]

    global Markers, Back_T, Back_B, Shoulder_R, Shoulder_L, Elbow_R, Elbow_L, Wrist_R, Wrist_L, Hip_R, Hip_L, Knee_R, Knee_L, Ankle_R, Ankle_L, Heel_R, Heel_L, Toe_R, Toe_L

    Back_T, Back_B = Back_T_var.get(), Back_B_var.get()
    Shoulder_R, Shoulder_L = Shoulder_R_var.get(), Shoulder_L_var.get()
    Elbow_R, Elbow_L = Elbow_R_var.get(), Elbow_L_var.get()
    Wrist_R, Wrist_L = Wrist_R_var.get(), Wrist_L_var.get()
    Hip_R, Hip_L = Hip_R_var.get(), Hip_L_var.get()
    Knee_R, Knee_L = Knee_R_var.get(), Knee_L_var.get()
    Ankle_R, Ankle_L = Ankle_R_var.get(), Ankle_L_var.get()
    Heel_R, Heel_L = Heel_R_var.get(), Heel_L_var.get()
    Toe_R, Toe_L = Toe_R_var.get(), Toe_L_var.get()

    Markers = [Back_T, Back_B, Shoulder_R, Shoulder_L, Elbow_R, Elbow_L, Wrist_R,
               Wrist_L, Hip_R, Hip_L, Knee_R, Knee_L, Ankle_R, Ankle_L, Heel_R,
               Heel_L, Toe_R, Toe_L]

    global direction, unnumerate_events_flag
    direction = direction_var.get()
    unnumerate_events_flag = unnumerate_var.get()

    window.destroy()
    showinfo(title="Информация", message="Все успешно сохранено", icon=INFO, default=OK)
    if path_din_tf.get():
        m.load_model(path, key_moments)
        m_events = m.__events__.name.to_list()
        if m_events[-1][-1] not in '0123456789':
            m.numerate_events()
        m.strip_data()
        steps = list(range(1, m.num_steps + 1))
        steps_from_cb["state"] = NORMAL
        steps_to_cb["state"] = NORMAL
        steps_from_cb["value"] = steps
        steps_from_cb.set('')
        steps_from_cb.insert(0, steps[0])
        steps_to_cb["value"] = steps
        steps_to_cb.set('')
        steps_to_cb.insert(0, steps[-1])


def reset_data():
    global key_moments, Markers, direction, unnumerate_events_flag
    global fs, ms, to, fs_var, ms_var, to_var
    global Back_T_var, Back_B_var, Shoulder_R_var, Shoulder_L_var, Elbow_R_var, Elbow_L_var, Wrist_R_var, Wrist_L_var, Hip_R_var, Hip_L_var, Knee_R_var, Knee_L_var, Ankle_R_var, Ankle_L_var, Heel_R_var, Heel_L_var, Toe_R_var, Toe_L_var
    global direction_var, unnumerate_var
    global Back_T, Back_B, Shoulder_R, Shoulder_L, Elbow_R, Elbow_L, Wrist_R, Wrist_L, Hip_R, Hip_L, Knee_R, Knee_L, Ankle_R, Ankle_L, Heel_R, Heel_L, Toe_R, Toe_L
    # Ключевые моменты
    fs_var.set(value="fs")
    ms_var.set(value="ms")
    to_var.set(value="to")

    # Маркеры
    Back_T_var.set(value='Back_T'),
    Back_B_var.set(value='Back_B')
    Shoulder_R_var.set(value='Shoulder_R')
    Shoulder_L_var.set(value='Shoulder_L')
    Elbow_R_var.set(value='Elbow_R')
    Elbow_L_var.set(value='Elbow_L')
    Wrist_R_var.set(value='Wrist_R')
    Wrist_L_var.set(value='Wrist_L')
    Hip_R_var.set(value='Hip_R')
    Hip_L_var.set(value='Hip_L')
    Knee_R_var.set(value='Knee_R')
    Knee_L_var.set(value='Knee_L')
    Ankle_R_var.set(value='Ankle_R')
    Ankle_L_var.set(value='Ankle_L')
    Heel_R_var.set(value='Heel_R')
    Heel_L_var.set(value='Heel_L')
    Toe_R_var.set(value='Toe_R')
    Toe_L_var.set(value='Toe_L')

    # Направление
    direction_var.set(value="-x")

    unnumerate_var.set(value="Да")

    fs = fs_var.get()
    ms = ms_var.get()
    to = to_var.get()

    Back_T, Back_B = Back_T_var.get(), Back_B_var.get()
    Shoulder_R, Shoulder_L = Shoulder_R_var.get(), Shoulder_L_var.get()
    Elbow_R, Elbow_L = Elbow_R_var.get(), Elbow_L_var.get()
    Wrist_R, Wrist_L = Wrist_R_var.get(), Wrist_L_var.get()
    Hip_R, Hip_L = Hip_R_var.get(), Hip_L_var.get()
    Knee_R, Knee_L = Knee_R_var.get(), Knee_L_var.get()
    Ankle_R, Ankle_L = Ankle_R_var.get(), Ankle_L_var.get()
    Heel_R, Heel_L = Heel_R_var.get(), Heel_L_var.get()
    Toe_R, Toe_L = Toe_R_var.get(), Toe_L_var.get()

    key_moments = [fs, ms, to]
    Markers = [Back_T, Back_B, Shoulder_R, Shoulder_L, Elbow_R, Elbow_L, Wrist_R,
               Wrist_L, Hip_R, Hip_L, Knee_R, Knee_L, Ankle_R, Ankle_L, Heel_R,
               Heel_L, Toe_R, Toe_L]
    direction = direction_var.get()
    unnumerate_events_flag = unnumerate_var.get()


frame_calculations = Frame(
    notebook
)

frame_graphics = Frame(
    notebook
)


canvas = Canvas(frame_graphics, width=1400, height=650)
canvas.pack(anchor=CENTER, expand=1)
canvas.create_rectangle(0, 0, 162, 650)


def setting():
    global frame_markers

    frame_markers = Toplevel(root)
    frame_markers.title("Названия маркеров и кл. моментов")
    frame_markers.geometry("800x500+380+150")
    key_moments_lb = Label(frame_markers, text="Ключевые моменты:")
    fs_lb = Label(frame_markers, text="Момент первого касания опоры")
    ms_lb = Label(frame_markers, text="Момент вертикали")
    to_lb = Label(frame_markers, text="Момент завершения отталкивания")
    key_moments_lb.place(relx=0.1, rely=0.23)
    fs_lb.place(relx=0.02, rely=0.3)
    ms_lb.place(relx=0.02, rely=0.35)
    to_lb.place(relx=0.02, rely=0.4)

    fs_tf = ttk.Entry(
        frame_markers,
        width=10,
        textvariable=fs_var
    )
    fs_tf.place(relx=0.27, rely=0.3)


    ms_tf = ttk.Entry(
        frame_markers,
        width=10,
        textvariable=ms_var
    )
    ms_tf.place(relx=0.27, rely=0.35)


    to_tf = ttk.Entry(
        frame_markers,
        width=10,
        textvariable=to_var
    )
    to_tf.place(relx=0.27, rely=0.4)


    # Маркеры
    markers_lb = Label(frame_markers, text="Маркеры:")
    markers_lb.place(relx=0.59, rely=0.23)

    # Части спины
    part_back_lb = Label(frame_markers, text="Верхняя                               Нижняя")
    part_back_lb.place(relx=0.6, rely=0.28)

    Back_lb = Label(frame_markers, text="Части спины:")
    Back_lb.place(relx=0.4, rely=0.28)

    Back_T_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Back_T_var
    )

    Back_B_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Back_B_var
    )
    Back_T_tf.place(relx=0.67, rely=0.28)
    Back_B_tf.place(relx=0.84, rely=0.28)

    # Плечевые суставы
    r_or_l_shoulder_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_shoulder_lb.place(relx=0.6, rely=0.33)

    Shoulder_lb = Label(frame_markers, text="Плечевые суставы:")
    Shoulder_lb.place(relx=0.4, rely=0.33)

    Shoulder_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Shoulder_R_var
    )

    Shoulder_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Shoulder_L_var
    )
    Shoulder_R_tf.place(relx=0.67, rely=0.33)
    Shoulder_L_tf.place(relx=0.84, rely=0.33)

    # Локтевые суставы
    r_or_l_elbow_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_elbow_lb.place(relx=0.6, rely=0.38)

    Elbow_lb = Label(frame_markers, text="Локтевые суставы:")
    Elbow_lb.place(relx=0.4, rely=0.38)

    Elbow_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Elbow_R_var
    )

    Elbow_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Elbow_L_var
    )
    Elbow_R_tf.place(relx=0.67, rely=0.38)
    Elbow_L_tf.place(relx=0.84, rely=0.38)

    # Лучезапястные суставы
    r_or_l_wrist_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_wrist_lb.place(relx=0.6, rely=0.43)

    Wrist_lb = Label(frame_markers, text="Лучезапястные суставы:")
    Wrist_lb.place(relx=0.4, rely=0.43)

    Wrist_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Wrist_R_var
    )

    Wrist_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Wrist_L_var
    )
    Wrist_R_tf.place(relx=0.67, rely=0.43)
    Wrist_L_tf.place(relx=0.84, rely=0.43)

    # Тазобедренные суставы
    r_or_l_hip_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_hip_lb.place(relx=0.6, rely=0.48)

    Hip_lb = Label(frame_markers, text="Тазобедренные суставы:")
    Hip_lb.place(relx=0.4, rely=0.48)

    Hip_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Hip_R_var
    )

    Hip_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Hip_L_var
    )
    Hip_R_tf.place(relx=0.67, rely=0.48)
    Hip_L_tf.place(relx=0.84, rely=0.48)

    # Коленные суставы
    r_or_l_knee_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_knee_lb.place(relx=0.6, rely=0.53)

    Knee_lb = Label(frame_markers, text="Коленные суставы:")
    Knee_lb.place(relx=0.4, rely=0.53)

    Knee_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Knee_R_var
    )

    Knee_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Knee_L_var
    )
    Knee_R_tf.place(relx=0.67, rely=0.53)
    Knee_L_tf.place(relx=0.84, rely=0.53)

    # Голеностопные суставы
    r_or_l_ankle_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_ankle_lb.place(relx=0.6, rely=0.58)

    Ankle_lb = Label(frame_markers, text="Голеностопные суставы:")
    Ankle_lb.place(relx=0.4, rely=0.58)

    Ankle_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Ankle_R_var
    )

    Ankle_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Ankle_L_var
    )
    Ankle_R_tf.place(relx=0.67, rely=0.58)
    Ankle_L_tf.place(relx=0.84, rely=0.58)

    # Пятки
    r_or_l_heel_lb = Label(frame_markers, text="Правая                                 Левая")
    r_or_l_heel_lb.place(relx=0.6, rely=0.63)

    Heel_lb = Label(frame_markers, text="Пятки:")
    Heel_lb.place(relx=0.4, rely=0.63)

    Heel_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Heel_R_var
    )

    Heel_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Heel_L_var
    )
    Heel_R_tf.place(relx=0.67, rely=0.63)
    Heel_L_tf.place(relx=0.84, rely=0.63)

    # Носки
    r_or_l_toe_lb = Label(frame_markers, text="Правый                                Левый")
    r_or_l_toe_lb.place(relx=0.6, rely=0.68)

    Toe_lb = Label(frame_markers, text="Носки:")
    Toe_lb.place(relx=0.4, rely=0.68)

    Toe_R_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Toe_R_var
    )

    Toe_L_tf = ttk.Entry(
        frame_markers,
        width=12,
        textvariable=Toe_L_var
    )
    Toe_R_tf.place(relx=0.67, rely=0.68)
    Toe_L_tf.place(relx=0.84, rely=0.68)

    # Направление
    markers_lb = Label(frame_markers, text="Направление:")
    markers_lb.place(relx=0.11, rely=0.53)

    direction_tf = ttk.Entry(
        frame_markers,
        width=7,
        textvariable=direction_var
    )
    direction_tf.place(relx=0.13, rely=0.58)

    unnumerate_events_lb = ttk.Label(frame_markers, text="Сбросить нумерацию ключевых моментов")
    unnumerate_events_lb.place(relx=0.03, rely=0.75)

    unnumerate_events_cb = ttk.Combobox(frame_markers, values=["Да", "Нет"], width=8, textvariable=unnumerate_var)
    unnumerate_events_cb.place(relx=0.14, rely=0.805)

    save_data_btn = ttk.Button(frame_markers, text="Сохранить", command=lambda: save_data(frame_markers))
    save_data_btn.place(relx=0.5, rely=0.9)

    reset_data_btn = ttk.Button(frame_markers, text="Сброс", command=reset_data)
    reset_data_btn.place(relx=0.4, rely=0.9)


import xlrd, xlwt


def print_mean_std(name, value, units, alt_fac=None, alt_uni=None):
    global line, pid
    ws.write(line, 0, pid)
    ws.write(line, 1, name)
    ws.write(line, 3, round(np.mean(value), 2))
    ws.write(line, 4, round(np.std(value), 2))
    ws.write(line, 5, units)
    if alt_fac:
        ws.write(line, 6, round(np.mean(value) / alt_fac * 100, 2))
        ws.write(line, 7, round(np.std(value) / alt_fac * 100, 2))
        ws.write(line, 8, alt_uni)
    line += 1
    pid += 1


def print_av_min_max(name, av, mi, ma, units):
    global line, pid
    ws.write(line, 0, pid)
    ws.write(line, 1, name)
    ws.write(line, 2, 'average')
    ws.write(line + 1, 2, 'min')
    ws.write(line + 2, 2, 'max')
    ws.write(line, 3, round(np.mean(av), 2))
    ws.write(line + 1, 3, round(np.mean(mi), 2))
    ws.write(line + 2, 3, round(np.mean(ma), 2))
    ws.write(line, 4, round(np.std(av), 2))
    ws.write(line + 1, 4, round(np.std(mi), 2))
    ws.write(line + 2, 4, round(np.std(ma), 2))
    ws.write(line, 5, units)
    ws.write(line + 1, 5, units)
    ws.write(line + 2, 5, units)
    line += 3
    pid += 1


def print_R_L(name, val_R, val_L, units, alt_fac=None, alt_uni=None):
    global line, pid
    ws.write(line, 0, pid)
    ws.write(line, 1, name)
    ws.write(line, 2, 'R')
    ws.write(line + 1, 2, 'L')
    ws.write(line + 2, 2, '(R+L)/2')
    ws.write(line + 3, 2, '(R-L)/[(R+L)/2]')
    ws.write(line, 3, round(np.mean(val_R), 2))
    ws.write(line + 1, 3, round(np.mean(val_L), 2))
    ws.write(line + 2, 3, round(np.mean(val_R + val_L) * .5, 2))
    ws.write(line + 3, 3, round(np.mean((val_R-val_L) / (val_R + val_L) / 2) * 100, 2))
    ws.write(line, 4, round(np.std(val_R), 2))
    ws.write(line + 1, 4, round(np.std(val_L), 2))
    ws.write(line + 2, 4, round(np.std((val_R + val_L) * .5), 2))
    ws.write(line + 3, 4, round(np.std((val_R-val_L) / (val_R + val_L) * 2) * 100, 2))
    ws.write(line, 5, units)
    ws.write(line + 1, 5, units)
    ws.write(line + 2, 5, units)
    ws.write(line + 3, 5, '%')
    if alt_fac:
        val_R /= alt_fac
        val_L /= alt_fac
        ws.write(line, 6, round(np.mean(val_R), 2))
        ws.write(line + 1, 6, round(np.mean(val_L), 2))
        ws.write(line + 2, 6, round(np.mean(val_R + val_L) * .5, 2))
        ws.write(line + 3, 6, round(np.mean(val_R / val_L) * 100, 2))
        ws.write(line, 7, round(np.std(val_R), 2))
        ws.write(line + 1, 7, round(np.std(val_L), 2))
        ws.write(line + 2, 7, round(np.std((val_R + val_L) * .5), 2))
        ws.write(line + 3, 7, round(np.std((val_R-val_L) / (val_R + val_L) * 2) * 100, 2))
        ws.write(line, 8, alt_uni)
        ws.write(line + 1, 8, alt_uni)
        ws.write(line + 2, 8, alt_uni)
        ws.write(line + 3, 8, '%')
    line += 4
    pid += 1


def export_xls():
    global line, pid
    line += 1
    pid = 8
    print_mean_std('Частота шагов (двойных)', 1 / step_duration_np, 'с-1')
    print_mean_std('Длительность шага (двойного)', step_duration_np, 'с')
    print_mean_std('Длина двойного шага', step_duration_np * speed / 3.6 * 100, 'см', height, '%H')
    print_mean_std('Длительность фазы опоры', stance_phase_np, 'с', T_step, '%T')
    if (type_move.get() == 'Бег'):
        print_mean_std('Длительность фазы полета', fly_phase_np, 'с', T_step, '%T')
    elif (type_move.get() == 'Ходьба'):
        print_R_L('Длительность двуопорного периода', double_contact_phase_R_np, double_contact_phase_L_np, 'с', T_step, '%T')
        print_R_L('Длительность фазы переноса', single_contact_phase_L_np, single_contact_phase_R_np, 'с', T_step, '%T')
    print_mean_std('Длительность фазы амортизации', depreciation_phase_np, 'с', T_step, '%T')
    print_mean_std('Длительность фазы отталкивания', repulsion_phase_np, 'с', T_step, '%T')
    line += 1
    ws.write(line, 1, 'Движение корпуса')
    line += 1
    print_mean_std('Размах вертикальных колебаний', vertical_swing_np, 'см', height, '%H')
    print_av_min_max('Угол наклона корпуса вперед-назад', sr_xz_np, mi_xz_np, ma_xz_np, 'град')
    print_R_L('Максимальный угол наклона корпуса вправо-влево', max_R, max_L, 'град')
    #print_R_L('Амплитуда вращения таза (max - min)', hip_ampl_R_np, hip_ampl_L_np, 'град')
    print_mean_std('Амплитуда вращения таза (max - min)', hip_ampl_np, 'град')
    #print_R_L('Амплитуда вращения плечевого пояса (max - min)', shoulder_ampl_R_np, shoulder_ampl_L_np, 'град')
    print_mean_std('Амплитуда вращения плечевого пояса (max - min)', shoulder_ampl_np, 'град')
    print_mean_std('Максимальный угол скручиваня (плечи-бедра)', swing_ampl_np, 'град')
    line += 1
    ws.write(line, 1, 'Движение рук')
    line += 1
    print_R_L('Вертикальное перемещение кисти (max - min)', ampl_R_np/10, ampl_L_np/10, 'cм', height, '%H')
    if (type_move.get() == 'Бег'):
        print_av_min_max('Угол в локтевом суставе R', mean_ang_elbow_right_np, min_ang_elbow_right_np,
                     max_ang_elbow_right_np, 'град')
        print_av_min_max('Угол в локтевом суставе L', mean_ang_elbow_left_np, min_ang_elbow_left_np, max_ang_elbow_left_np,
                     'град')
    elif (type_move.get() == 'Ходьба'):
        print_av_min_max('Угол сгибания в локте R', 180 - mean_ang_elbow_right_np, 180 - max_ang_elbow_right_np,
                     180 - min_ang_elbow_right_np, 'град')
        print_av_min_max('Угол сгибания в локте L', 180 - mean_ang_elbow_left_np, 180 - max_ang_elbow_left_np, 180 - min_ang_elbow_left_np,
                     'град')
    print_R_L('Максимальный угол сгибания плеча', should_flex_ang_max_R_np, should_flex_ang_max_L_np, 'град')
    print_R_L('Максимальный угол разгибания плеча', should_ext_ang_max_R_np, should_ext_ang_max_L_np, 'град')
    print_R_L('Максимальный угол отведения плеча', should_abd_ang_max_R_np, should_abd_ang_max_L_np, 'град')
    print_R_L('Длина траектории кисти', wrist_dist_r_np / 10, wrist_dist_l_np / 10, 'см')
    line += 1
    ws.write(line, 1, 'Движение маховой ноги')
    line += 1
    print_R_L('Минимальный угол складывания голени (угол в коленном суставе)', min_ang_ankle_r_np, min_ang_ankle_l_np,
              'град')
    print_R_L('Максимальный угол выноса бедра', max_hip_ext_ang_R_np, max_hip_ext_ang_L_np, 'град')
    line += 1
    ws.write(line, 1, 'Движение опорной ноги при постановке на опору')
    line += 1
    
    print_R_L('Угол разгбания стопы', ang_ankle_fs_r_np, ang_ankle_fs_l_np, 'град')
    
    if (type_move.get() == 'Бег'):
        print_R_L('Угол в коленном суставе', ang_knee_fs_r_np, ang_knee_fs_l_np, 'град')
    elif (type_move.get() == 'Ходьба'):
        print_R_L('Угол сгибания в колене', 180 - ang_knee_fs_r_np, 180 - ang_knee_fs_l_np, 'град')
    print_R_L('Угол в тазобедренном суставе', ang_hip_fs_r_np, ang_hip_fs_l_np, 'град')
    print_R_L('Вынос стопы', takeaway_step_fs_r_np, takeaway_step_fs_l_np, 'см', height, '%H')
    print_R_L('Угол выноса бедра (с вертикалью)', ang_bed_vert_r_np, ang_bed_vert_l_np, 'град')
    print_R_L('Угол постановки (голень-опора)', ang_gol_op_fs_r_np, ang_gol_op_fs_l_np, 'град')
    line += 1
    ws.write(line, 1, 'Движение опорной ноги при прохождении вертикали')
    line += 1
    print_R_L('Угол в коленном суставе', ang_knee_ms_r_np, ang_knee_ms_l_np, 'град')
    print_R_L('Угол в тазобедренном суставе', ang_hip_ms_r_np, ang_hip_ms_l_np, 'град')
    print_R_L('Высота пятки', heel_height_ms_r_np, heel_height_ms_l_np, 'см', height, '%H')
    line += 1
    ws.write(line, 1, 'Движение опорной ноги при завершении отталкивания')
    line += 1
    print_R_L('Угол в коленном суставе', ang_knee_to_r_np, ang_knee_to_l_np, 'град')
    print_R_L('Изменение угола в голеностопном суставе от статичного (угол разгбания)', ang_ankle_to_r_np, ang_ankle_to_l_np, 'град') 
    print_R_L('Угол в тазобедренном суставе', ang_hip_back_to_r_np, ang_hip_back_to_l_np, 'град')
    print_R_L('Угол бедра с вертикалью', ang_hip_vert_to_r_np, ang_hip_vert_to_l_np, 'град')
    print_R_L('Угол отталкивания (голень-опора)', ang_gol_op_to_r_np, ang_gol_op_to_l_np, 'град')
    wb.save(export_save_tf.get())
    showinfo(title="Информация", message="Успешно", icon=INFO, default=OK)


frame_export = Frame(notebook)
export_lb = Label(frame_export, text="Сохраните данные в excel файл", font=("Arial Bold", 17))
export_lb.place(relx=0.35, rely=0.25)
export_save_lb = Label(
    frame_export,
    text="Сохранить в .xls",
)
export_save_lb.place(relx=0.20, rely=0.4)
export_save_tf = ttk.Entry(
    frame_export,
    width=80
)
export_save_tf.place(relx=0.30, rely=0.397)
btn_for_save_xls_file = ttk.Button(
    frame_export,
    text='Обзор',
    command=save_xls_file
)
btn_for_save_xls_file.place(relx=0.67, rely=0.395)

# export_save_lb_round = Label(
#     frame_export,
#     text="Количество знаков после запятой:",
# )
# export_save_lb_round.place(relx=0.4, rely=0.65)

btn_export = ttk.Button(frame_export, text="Экспортировать", command=export_xls)
btn_export.place(relx=0.46, rely=0.55)

ttk.Button(notebook, image=photo, command=setting).place(relx=0.935, rely=0.85)

picture_lb2 = ttk.Label(notebook, image=photo2)
picture_lb2.place(relx=0.9, rely=0.0)

version_lb = ttk.Label(notebook, text = 'ver ' + version, font=("Arial Bold", 12))
version_lb.place(relx=0.01, rely=0.95)

notebook.add(frame, text="Данные")
notebook.add(frame_parameters, text="Параметры", state=DISABLED)
notebook.add(frame_calculations, text='Предпросмотр', state=DISABLED)
notebook.add(frame_graphics, text='Графики', state=DISABLED)
notebook.add(frame_export, text='Отчет', state=DISABLED)

root.mainloop()