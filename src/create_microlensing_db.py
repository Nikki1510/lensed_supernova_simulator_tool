import sqlite3
import json
import numpy as np
import time
from tqdm import tqdm


def load_lightcurve(kappa, gamma, s, source_redshift):
    """
    Load microlensing light curve dictionary.

    :param kappa: convergence (float)
    :param gamma: shear (float)
    :param s: smooth matter fraction (float)
    :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts (float)
    :return: d_light_curves: dictionary containing microlensing + macrolensing curves and time sampling
             length of the longest array in d_light_curves --> this will be the number of data columns in the database
    """

    # Directory containing the light curves
    output_data_path = "../data/microlensing/light_curves/"

    # Light curve properties
    N_sim = 10000
    lens_redshift = 0.32

    # Open corresponding pickle file with light curve
    pickel_name = "k%f_g%f_s%.3f_redshift_source_%.3f_lens%.3f_Nsim_%i" % (kappa, gamma, s, source_redshift,
                                                                           lens_redshift, N_sim)

    open_pickle = "%s%s.pickle" % (output_data_path, pickel_name)
    with open(open_pickle, 'r') as handle:
        d_light_curves = json.load(handle, encoding='latin1')

    return d_light_curves, len(d_light_curves['time_bin_center'])


def create_database(name):
    """
    Create a new, empty database.

    :param name: name of the database
    :return: an empty database has been created
    """

    conn = sqlite3.connect(name)

    c = conn.cursor()

    c.execute("""CREATE TABLE datapoints (
                filename    text,
                line_id     integer,
                data_type   text,
                bandpass    text,
                SN_model    text,
                data00       real,
                data01       real,
                data02       real,
                data03       real,
                data04       real,
                data05       real,
                data06       real,
                data07       real,
                data08       real,
                data09       real,
                data10       real,
                data11       real,
                data12       real,
                data13       real,
                data14       real,
                data15       real,
                data16       real,
                data17       real,
                data18       real,
                data19       real,
                data20       real,
                data21       real,
                data22       real,
                data23       real,
                data24       real,
                data25       real,
                data26       real,
                data27       real,
                data28       real,
                data29       real,
                data30       real,
                data31       real,
                data32       real,
                data33       real,
                data34       real,
                data35       real,
                data36       real,
                data37       real
                )""")

    c.execute("""CREATE UNIQUE INDEX index1_datapoints
                 ON datapoints (filename, line_id, data_type, bandpass, SN_model);
                """)
    conn.commit()
    conn.close()


def clear_database(name):
    """
    Remove everything inside the database.

    :param name: name of the database
    :return: the database is now completely empty
    """

    # connect to the database and get a cursor
    conn = sqlite3.connect(name)
    c = conn.cursor()

    # delete all rows from table
    print("Clear database")
    c.execute('DELETE FROM datapoints;', )
    conn.commit()


def count_nof_records(name):
    """
    Count the number of rows of the database.

    :param name: name of the database
    :return: number of rows the database contains (int)
    """

    # connect to the database and get a cursor
    conn = sqlite3.connect(name)
    c = conn.cursor()

    c.execute('SELECT COUNT(*) from datapoints')
    nof_rows = c.fetchone()
    print("Number of rows in table datapoints: ", nof_rows[0])


def write_to_database(kappa, gamma, s, source_redshift, name):
    """
    Write the entries of the pickle file corresponding to (kappa, gamma, s, z_src) to the database.

    :param kappa: convergence (float)
    :param gamma: shear (float)
    :param s: smooth matter fraction (float)
    :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts (float)
    :param name: name of the database
    :return: the database is filled with entries from the pickle file corresponding to (kappa, gamma, s, z_src)
    """

    # connect to the database and get a cursor
    conn = sqlite3.connect(name)
    c = conn.cursor()

    FIELD_LIST = """
        (:filename, :line_id, :data_type, :bandpass, :SN_model, :data00,
         :data01, :data02, :data03, :data04, :data05, :data06, :data07, :data08, :data09, :data10,
         :data11, :data12, :data13, :data14, :data15, :data16, :data17, :data18, :data19, :data20,
         :data21, :data22, :data23, :data24, :data25, :data26, :data27, :data28, :data29, :data30,
         :data31, :data32, :data33, :data34, :data35, :data36, :data37)
        """

    file_name = "kappa:%.3f_gamma:%.3f_s:%.3f_zsrc:%.2f" % (kappa, gamma, s, source_redshift)

    file, _ = load_lightcurve(kappa, gamma, s, source_redshift)
    tqdm._instances.clear()
    pbar = tqdm(total=len(file))

    for x in range(len(file)):

        # Extract the information about data_type, bandpass and SN_model from the key name
        key = list(file.keys())[x]
        data_type = key.split("_")[0]
        data = file[key]

        if data_type == 'micro' or 'macro':
            info = key.split("_")[-1]
            SN_model = info[0]
            bandpass = info[-1]
        else:
            SN_model = 'none'
            bandpass = 'none'

        row = {'filename': file_name,
               'line_id': x,
               'data_type': data_type,
               'bandpass': bandpass,
               'SN_model': SN_model,
               'data00': data[0],
               'data01': data[1],
               'data02': data[2],
               'data03': data[3],
               'data04': data[4],
               'data05': data[5],
               'data06': data[6],
               'data07': data[7],
               'data08': data[8],
               'data09': data[9],
               'data10': data[10],
               'data11': data[11],
               'data12': data[12],
               'data13': data[13],
               'data14': data[14],
               'data15': data[15],
               'data16': data[16],
               'data17': data[17],
               'data18': data[18],
               'data19': data[19],
               'data20': data[20],
               'data21': data[21],
               'data22': data[22],
               'data23': data[23],
               'data24': data[24],
               'data25': data[25],
               'data26': data[26],
               'data27': data[27],
               'data28': data[28],
               'data29': data[29],
               'data30': data[30],
               'data31': data[31],
               'data32': data[32],
               'data33': data[33],
               'data34': data[34],
               'data35': data[35],
               'data36': data[36],
               'data37': data[37]
               }

        c.execute("INSERT INTO datapoints VALUES " + FIELD_LIST, row)
        pbar.update(1)

    # commit the transaction
    conn.commit()


def fill_database(name, source_redshift):
    """
    Fill the database with entries from all files corresponding to the given source_redshift.
    This includes 16 combinations of (kappa, gamma, s), which are defined in 'kgs'.

    :param name: name of the database
    :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts (float)
    :return: the database for the given source_redshift is now completely full
    """

    count_nof_records(name)

    kgs = np.around([(3.620489580770965832e-01, 3.416429828804125046e-01, 4.430360463808165061e-01),
                     (6.550590438288885764e-01, 6.694697862208409678e-01, 4.430360463808165061e-01),
                     (6.550590438288885764e-01, 9.517424234642006819e-01, 4.430360463808165061e-01),
                     (9.564962670984238358e-01, 6.694697862208409678e-01, 4.430360463808165061e-01),
                     (9.564962670984238358e-01, 9.517424234642006819e-01, 4.430360463808165061e-01),
                     (3.620489580770965832e-01, 3.416429828804125046e-01, 0.616),
                     (6.550590438288885764e-01, 6.694697862208409678e-01, 0.616),
                     (6.550590438288885764e-01, 9.517424234642006819e-01, 0.616),
                     (9.564962670984238358e-01, 6.694697862208409678e-01, 0.616),
                     (9.564962670984238358e-01, 9.517424234642006819e-01, 0.616),
                     (3.620489580770965832e-01, 3.416429828804125046e-01, 7.902578031429109418e-01),
                     (6.550590438288885764e-01, 6.694697862208409678e-01, 7.902578031429109418e-01),
                     (6.550590438288885764e-01, 9.517424234642006819e-01, 7.902578031429109418e-01),
                     (9.564962670984238358e-01, 6.694697862208409678e-01, 7.902578031429109418e-01),
                     (9.564962670984238358e-01, 9.517424234642006819e-01, 7.902578031429109418e-01),
                     (3.620489580770965832e-01, 2.800000000000000266e-01, 9.100000000000000311e-01)], 3)

    for K, G, S in kgs:
        print("Kappa, gamma, s = ", K, G, S)
        start = time.time()
        write_to_database(kappa=K, gamma=G, s=S, source_redshift=source_redshift, name=name)
        finish = time.time()
        duration = finish - start
        print("Duration: ", np.around(duration / 60, 2), " minutes")
        count_nof_records(name)
        print(" ")

    print("--- Done!! ---")
    count_nof_records(name)


def query_database(name, kappa, gamma, s, source_redshift, data_type, bandpass, SN_model):
    """
    Select one element (corresponding to an array) from the database.

    :param name: name of the database
    :param kappa: convergence (float)
    :param gamma: shear (float)
    :param s: smooth matter fraction (float)
    :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts (float)
    :param data_type: this refers to either a microlensing light curve ('micro'), a macrolensing light curve ('macro'),
           or the time sampling of the light curves ('time')
    :param bandpass: telescope filter used for simulation. choose from ["u","g","r","i","z","y","J","H"]
    :param SN_model: supernova explosion model used in the simulation. choose from ["m", "n", "w", "s"]
    :return: single entry (array) from the database, either a micro light curve, macro light curve, or time sampling
    """

    file_name = "kappa:%.3f_gamma:%.3f_s:%.3f_zsrc:%.2f" % (kappa, gamma, s, source_redshift)

    # connect to the database and get a cursor
    conn = sqlite3.connect(name)
    c = conn.cursor()

    condition = "filename = :filename AND data_type = :data_type AND bandpass = :bandpass AND SN_model = :SN_model"
    condition_values = {'filename': file_name,
                        'data_type': data_type,
                        'bandpass': bandpass,
                        'SN_model': SN_model}

    c.execute("SELECT * FROM datapoints WHERE " + condition + " ORDER BY RANDOM() LIMIT 1",
              condition_values)

    result = c.fetchone()
    print(result)

    return result


def get_microlensing(name, kappa, gamma, s, source_redshift, bandpass):
    """
    Obtain the microlensing contribution (= difference between macro and micro light curve), macrolensing curve,
    and time sampling for a particular lensed supernova system.

    :param name: database name
    :param kappa: convergence (float)
    :param gamma: shear (float)
    :param s: smooth matter fraction (float)
    :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts (float)
    :param bandpass: telescope filter used for simulation. choose from ["u","g","r","i","z","y","J","H"]
    :return: 3 arrays containing the microlensing contributions, macrolensing light curve, and  time sampling
    """

    SN_model = np.random.choice(["m", "n", "w", "s"])
    if source_redshift == 1.4:
        SN_model = "w"

    micro = query_database(name, kappa, gamma, s, source_redshift, 'micro', bandpass, SN_model)[5:]
    macro = query_database(name, kappa, gamma, s, source_redshift, 'macro', bandpass, SN_model)[5:]
    time_range = query_database(name, kappa, gamma, s, source_redshift, 'time', 'none', 'none')[5:]

    micro_contribution = np.array(micro) - np.array(macro)

    return micro_contribution, np.array(macro), np.array(time_range)


def main():

    """
    # Fill database
    
    print("Redshift = 1.05")
    database_name = 'microlensing_database_z_1_05.db'
    create_database(name=database_name)
    count_nof_records(database_name)
    fill_database(name=database_name, source_redshift=1.05)

    
    # Change input for SN_model and bandpass to 'none' for all timeseries 
    
    z_list = np.arange(0, 1.45, 0.05)

    for z in z_list:

        database_name = '../data/microlensing/databases/microlensing_database_z_%i_%s.db' % (int(np.floor(z)),
                                 np.char.zfill(str(np.around(100 * (z - int(np.floor(z))))), 2))
        print(database_name)
    
        # connect to the database and get a cursor
        conn = sqlite3.connect(database_name)
        c = conn.cursor()
    
        c.execute("UPDATE datapoints SET SN_model =:SN_model WHERE data_type =:data_type",
                  {'SN_model': 'none',
                   'data_type': 'time'}
                  )
    
        c.execute("UPDATE datapoints SET bandpass =:bandpass WHERE data_type =:data_type",
                  {'bandpass': 'none',
                   'data_type': 'time'}
                  )
    
        # commit the transaction
        conn.commit()

        
    """



if __name__ == '__main__':
    main()

    z_src = 1.05

    # database_name should match filename of the database of interest (dependent on source redshift)
    database_name = '../data/microlensing/databases/microlensing_database_z_%i_%s.db' % \
                    (int(np.floor(z_src)), np.char.zfill(str(int(np.around(100 * (z_src - int(np.floor(z_src)))))), 2))

    # print(database_name)

    # micro, macro, time_range = get_microlensing(database_name, kappa=0.362, gamma=0.28, s=0.910,
    #                                            source_redshift=z_src, bandpass='i')

    print(" --------------- ")

    kappa, gamma, s = 0.362000, 0.280000, 0.910
    source_redshift = 0.5

    file_name = "kappa:%.3f_gamma:%.3f_s:%.3f_zsrc:%.2f" % (kappa, gamma, s, source_redshift)
    print("File name: ", file_name)

    file, _ = load_lightcurve(kappa, gamma, s, source_redshift)

    for x in range(10):
        # Extract the information about data_type, bandpass and SN_model from the key name
        key = list(file.keys())[x]
        data_type = key.split("_")[0]
        data = file[key]

        info = key.split("_")[-1]
        SN_model = info[0]
        bandpass = info[-1]

        print(data_type, SN_model, bandpass, data)

    """
    

    kappa, gamma, s = 3.620489580770965832e-01, 3.416429828804125046e-01, 4.430360463808165061e-01
    source_redshift = 0.5

    file_name = "kappa:%.3f_gamma:%.3f_s:%.3f_zsrc:%.2f" % (kappa, gamma, s, source_redshift)
    print("File name: ", file_name)


    """
