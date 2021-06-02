import pandas as pd
import numpy as np
from scipy import stats
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import f1_score

np.random.seed(0)

class SyMProD:
    """
    This class is called Synthetic Minority Probabilistic Distribution Oversampling technique,
    whole class and its methods are based by this article https://sci-hub.do/10.1109/ACCESS.2020.3003346,
    Algorithm is divided by below steps.

    Step1: Find number of instances that has to be generated for balancing data

    Step2: Get out of outliers in minority instances by given noise threshold with Z_score method, and find k the
    nearest minority and majority neighbors for each minority instance

    Step3: Get out from overlapping instances by given cutoff threshold

    Step4: Find the m nearest minority instances for each remained minority instance

    Step5: Generate new instances with Probabilistic method
    """

    def __init__(self, nt: float, ct: float, k: int, m: int) -> None:
        """
        :param nt: noise threshold of getting out from outliers
        :param ct: cutoff threshold of for getting out from overlapping instances
        :param k: number of nearest neighbors for overlapping
        :param m: number of nearest neighbors for generating new instances


        """
        self.nt = nt
        self.ct = ct
        self.k = k
        self.m = m
        self.summary = None

    def dist_matrix(self, x: np.array, x_max=None) -> np.array:
        """
        Return distance matrix size=(x.shape[0], x.shape[0]-1), diagonal is not included as they are 0s,
        each row element show the euclidean distance between instances.

        Conditions: if x_max is not given this will return distance matrix of x,

        Otherwise this will return distance matrix between x and x_max

        :param x: np.array like np.array, pd.DataFrame, list, nested list, instances
        :param x_max: np.array like np.array, pd.DataFrame, list, nested list, instances of majority class
        :return: dist matrix
        """
        if x_max is not None:  # case when x_max is given
            x = np.array(x)
            x_max = np.array(x_max)  # making arrays np.array for indexing and slicing simplicity
            res = np.zeros((x.shape[0], x_max.shape[0]))
            for j in range(x.shape[0]):
                # making the size of (x_max instances * num of x columns) matrix with same values subtract by
                # x_max instances and then calculate norm of rows
                res[j, :] = np.linalg.norm(np.ones((x_max.shape[0], x.shape[1])) * x[j, :] - x_max, axis=1)
            return res
        # case when x_max is not given
        x = np.array(x)
        res = np.zeros((x.shape[0], x.shape[0] - 1))  # columns are less by one element because we exclude diagonals
        res[0, :] = np.linalg.norm(np.ones((x.shape[0] - 1, x.shape[1])) * x[0, :] - x[1:, :], axis=1)  # first row
        res[-1, :] = np.linalg.norm(np.ones((x.shape[0] - 1, x.shape[1])) * x[-1, :] - x[:-1, :], axis=1)  # last row
        for i in range(1, res.shape[0] - 1):
            # inner rows are calculated with this method
            same = np.ones((x.shape[0] - 1, x.shape[1])) * x[i, :]
            others = np.vstack((x[:i, :], x[i + 1:, :]))  # concatenate rows of matrix without inner the ird instance
            res[i, :] = np.linalg.norm(same - others, axis=1)
        return res

    def outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame without outlier rows, with z_score method by given noise threshold
        the row is excluded if any column satisfied outlier condition

        :param df: dataframe with outliers
        :return: dataframe without outlier instances
        """
        return df[(np.abs(stats.zscore(df)) < self.nt).any(axis=1)]

    def taos(self, df_min: np.array, df_maj: np.array, s_min: np.array, s_maj: np.array)->tuple:
        """
        Return Taos of minority and majority instances for each minority instance

        :param df_min: minority instances
        :param df_maj: minority instances
        :param s_min:  matrix of k nearest minority neighbor's indexes in df_min for each minority instance
        :param s_maj:  matrix of k nearest majority neighbor's indexes in df_maj for each minority instance
        :return: tao arrays for each minority instance
        """
        taos_min, taos_maj = [], []
        for i in range(df_min.shape[0]):
            ind1, ind2 = s_min[i, :], s_maj[i, :]  # for each minority instance its k nearest min_maj neighbors indexes
            # closeness factor (minority and majority) for each instance
            c_min = sum(1 / (np.linalg.norm(df_min[ind1, :], axis=1)))
            c_maj = sum(1 / (np.linalg.norm(df_maj[ind2, :], axis=1)))

            # euclidean norm among each instance and its minority and majority neighbors
            dist_min = sum(
                np.linalg.norm((np.ones((self.k, df_min.shape[1])) * df_min[i, :]) - df_min[ind1, :], axis=1))
            dist_maj = sum(
                np.linalg.norm((np.ones((self.k, df_maj.shape[1])) * df_min[i, :]) - df_maj[ind2, :], axis=1))
            taos_min.append(c_min / dist_min)
            taos_maj.append(c_maj / dist_maj)
        return np.array(taos_min), np.array(taos_maj)

    def min_max_scale(self, data: pd.DataFrame or np.array, method: str) -> pd.DataFrame:
        """
        Return transformed scaled or inverse transformed data by min_max scaling method

        Condition: if method == 'transform' this will return scaled data,

        Otherwise: NOTE that data has to be already scaled for inversing it

        :param data: dataframe
        :param method: transform means to scale data by each column
        :return: dataframe
        """
        if method == 'transform':
            return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
        else:
            # in this case NOTE that data has to be scaled
            return (data * (np.max(data, axis=0) - np.min(data, axis=0))) + np.min(data, axis=0)

    def random_weights(self, p: np.array) -> np.array:
        """
        Return normalized weights by given probabilities and random generated weights

        :param p: probabilities as array
        :return: normalize weights
        """
        rand = np.random.randint(low=0, high=100, size=len(p))
        rand_norm = rand / rand.sum(0)
        #assert sum(rand_norm).round(decimals=0) == 1
        weights = rand_norm * p
        weights_norm = (weights / weights.sum(0))
        #assert sum(weights_norm).round(decimals=0) == 1
        return weights_norm

    def fit(self, data: pd.DataFrame, method: str, num_gen: int or None) -> pd.DataFrame or None:
        """
        Return Synthetically Balanced data

        Precondition1: data must not be scaled and

        Precondition2: Target column has to be last column in given data

        Disclaimer: ordinal and categorical features won't be synthesized properly

        :param data: unbalanced data shaped (n, c)
        :param num_gen: number of new generated instances,
                    if num_gen is None it will generate at least required instances for balancing
        :param method: the way that new instances will be chosen
        :return:  balanced data shaped(n + g, c)
        """
        # Notations used in comments
        # n - number of rows; g - number of generated rows; c - number of columns
        # k & m - number of neighbors
        # out1/out2 - number of outliers
        # over - number of overlapping rows

        # Step 1:
        # Find number of the rows that has to be generated, both minority and majority labels and submatrixes of data
        start = datetime.datetime.now()
        columns = data.columns
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        vals, counts = np.unique(y.values, return_counts=True)
        n_maj, n_min = max(counts), min(counts)
        n_gen = n_maj - n_min  # number of new instances
        if num_gen is None:
            num_gen = n_gen
        try:
            assert(n_gen <= num_gen)
        except AssertionError:
            print(f'Can not perform fit method, because given num_gen({num_gen})'
                  f' is lower than required new rows quantity({n_gen}),'
                  f' put num_gen variable at least {n_gen}')
            return None
        y_maj, y_min = vals[counts.argmax()], vals[counts.argmin()]  # minority and majority labels
        # subseting dataframe into 2 dataframes
        self.summary = {
            "Given data shape": data.shape, "Majority instance label and count": (y_maj, n_maj),
            "Minority instance label and count": (y_min, n_min), "Number of new instances": n_gen
        }


        df_maj, df_min = data.loc[data.iloc[:, -1] == y_maj].iloc[:, :-1], \
                         data.loc[data.iloc[:, -1] == y_min].iloc[:, :-1]

        # Step 2
        # Normalize data, get out of outliers by noise threshold and find nearest neighbors

        # keep scaled data for concatenating in last step (n, c), (n, c)
        df_maj_normed1, df_min_normed1 = self.min_max_scale(df_maj, 'transform'), \
                                         self.min_max_scale(df_min, 'transform')
        # get out of outliers (n-out1, c), (n-out2, c) out1/out2 number of outliers in min_maj subsets
        df_maj_out, df_min_out = self.outliers_zscore(df_maj).values, \
                                 self.outliers_zscore(df_min).values
        # scaling data without outliers (n-out1, c), (n-out2, c)
        # conditionally we will keep rows as _n_
        df_maj_normed, df_min_normed = self.min_max_scale(df_maj_out, 'transform'), \
                                       self.min_max_scale(df_min_out, 'transform')
        # get distance matrix for each minority instance (n, n-1), (n, n_maj) matrixes
        x_min_dist, x_minmaj_dist = self.dist_matrix(x=df_min_normed), \
                                    self.dist_matrix(x=df_min_normed, x_max=df_maj_normed)
        # get k nearest minority and majority instance indexes for each minority instance
        # (n, k), (n, k) matrixes
        s_min_ind, s_maj_ind = x_min_dist.argsort(axis=1)[:, :self.k], x_minmaj_dist.argsort(axis=1)[:, :self.k]

        # Step 3
        # Get Taos of each instance, exclude overlapping samples, calculate probability distribution of instance

        # get Taos of each instance (n, ) (n, ) vectors
        taos_min, taos_maj = self.taos(df_min_normed, df_maj_normed, s_min_ind, s_maj_ind)
        # get out overlapping instances by cutoff threshold(ct)
        # check np.where documentation

        ind_min = np.where(taos_min > taos_maj * self.ct)[0]
        # new instances without overlapping ones (n - over, c)
        df_min_new = df_min_normed[ind_min, :]
        # calculate phi and P dist for each instance (n-over, ) (n-over, ) arrays

        phi = (taos_min + 1) / (taos_maj + 1)
        P = phi / sum(phi)

        new_instances = np.zeros((num_gen, df_min_new.shape[1]))
        # nearest m neighbors of remained instances (n-over, c) matrix
        inds_min = self.dist_matrix(x=df_min_new).argsort(axis=1)[:, :self.m]
        # inserting minority indexes as first column in index matrixes (n-over, m+1)
        inds_min = np.insert(inds_min, 0, ind_min, axis=1)

        # Step 4
        # Generate the new instances by below strategy:
        # For each new instance
        # Randomly choose real instance and its m nearest neighbors,
        # Then randomly generate normalized weights for chosen m+1 instances
        # Finally the new instance will be weighted sum of that instances

        for i in range(num_gen):
            rand_ind = np.random.choice(len(ind_min))  # select random real instance index
            index = inds_min[rand_ind, :]  # taking that instance and its nearest neighbors indexes
            P_tmp = P[index]  # Probability dist values of that instances
            weights = self.random_weights(P_tmp)  # normalized weights for that instances

            # firstly create ones matrix shaped (c, m+1) multiple each row with normalized weights
            # then transpose the matrix (m+1, c) shaped, each row elements have same values
            same = (weights * np.ones(shape=(df_min_new.shape[1], inds_min.shape[1]))).T
            # multiplying element vise weights and chosen instances
            new_inst_matrix = same * df_min_normed[index, :]
            # the new instance is the weightes sum of chosen instances
            new_instances[i, :] = np.sum(new_inst_matrix, axis=0)

        # Step 5
        # Stack kept in Step2 minority and majority data rows with new instances
        # Finally create the balanced DataFrame

        best_inds = 0
        if method == 'random':
            best_inds = np.random.choice(np.arange(0, num_gen), n_gen, replace=True)
        elif method == 'near_cumm':
            dist_matrix_new_instances = self.dist_matrix(x=new_instances, x_max=df_min_new)
            dist_vector = dist_matrix_new_instances.sum(1)
            best_inds = dist_vector.argsort()[: n_gen]
        elif method == 'near_col':
            dist_matrix_new_instances = self.dist_matrix(x=new_instances, x_max=df_min_new)
            first_col = dist_matrix_new_instances[:, 0]
            best_inds = first_col.argsort()[:n_gen]

        best_instances = new_instances[best_inds, :]
        data = np.vstack((df_min_normed1, df_maj_normed1, best_instances))
        y_gen = np.ones(n_gen, dtype=np.int32) * y_min
        labels = np.append(y, y_gen, axis=0)
        data = pd.DataFrame(data=data)
        data.columns = columns[:-1]
        data[columns[-1]] = np.array(labels)
        finish = datetime.datetime.now()
        self.summary['Time taken'] = str(finish - start)
        self.summary["New data shape"] = data.shape
        data = data.sample(frac=1.0).copy()
        return data


if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\rafael_s\\Downloads\\telecom_churn.csv")
    df = df[list(df.columns[1:]) + [df.columns[0]]].copy()
    model = SyMProD(nt=3.0, ct=0.9, k=5, m=3)
    df_new = model.fit(data=df, method='random', num_gen=None)
    X, y = df_new.iloc[:, :-1], df_new.iloc[:, -1]
    x_train, x_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("The F1 score is  ", f1_score(y_test, y_pred))
    print()
    print(model.summary)


