#!/usr/bin/env python3
import re
import copy
import numpy as np
from   typing            import Iterator, Union, Callable, Tuple, List, Any
from   sklearn.neighbors import KernelDensity
from   scipy.special     import softmax

REPR_WIDTH = 100

def kde_sklearn(data: np.ndarray, bandwidth: float):
    """ Convenience wrapper for the sklearn.neighbors.KernelDensity. This algorithm can be used to identify clusters in a mass of points. Bandwith is a very important tuning parameter! Set it to roughly the radius of the clusters you want to identify.
    
    Args
    ----
    - `data`: datapoints in a numpy array of shape (n, d), where n is number of points and d is the dimensionality.
    - `bandwidth`: important parameter for controlling smoothing strength. set equal to radius of expected cluster size!
    
    Returns
    -------
    - `densities`: the density values at every input point. numpy array of shape (n, )
    - `densest_point`: coordinates of the point with the highest density. numpy array of shape (d, )
    """
    
    kde = KernelDensity(
        kernel    = "epanechnikov", # [gaussian, tophat, epanechnikov, exponential, linear, cosine] (not so important)
        bandwidth = bandwidth,      # important!
        algorithm = "auto", 
        rtol      = 1e-3, # speeds up algorithm by allowing some error
        atol      = 1e-3, # speeds up algorithm by allowing some error
        ).fit(data)
    
    log_densities = kde.score_samples(data)    # returns log densities
    densities     = np.exp(log_densities)      # convert log density to actual density
    densest_point = data[np.argmax(densities)]
    
    return np.array(densities), np.array(densest_point)

def imu_interpolation(points: np.ndarray):
    """ takes the last few imu measurements (either linear or angular velocity) and finds its principle axis of change (equivalent to linear regression in 3D) the last measurement point is then projected onto the linear interpolation to find the 'latest smoothed velocity estimate'
    
    Aargs
    -----
    - `points`: np.ndarray of shape (n, 3) with either linear or angular velocity measurements to be smoothed  
    """
    
    # center the points to their center of mass for SVD to work correctly
    center          = np.mean(points, axis=0)
    points_centered = points - center
    
    # perform SVD (numpy uses BLAS / LAPACK so this is pretty efficient)
    _, _, Vt = np.linalg.svd(points_centered, full_matrices=False, compute_uv=True)
    
    # get the principal direction as normal vector
    direction_norm = Vt[0] / np.linalg.norm(Vt[0], ord=2)
    
    # project the last point onto the principal direction (and re-offset by center!)
    projection = center + np.dot(points_centered[-1], direction_norm) * direction_norm
    
    return projection

class StructuredDeque:
    def __init__(self, maxlen: int, datatype: List[Tuple[str, np.dtype, Tuple]]):
        """ a class for a simple structured numpy array with deque-like behaviour.
        
        Args
        ----
        - `maxlen`: maximum length of the deque, after which elements are pushed off 
        - `datatype`: numpy datatype in the form of a list of tuples [("fieldname", np.dtype, shape_tuple), (...)]
        """
        
        self.MAXLEN = maxlen
        self.DTYPE  = datatype
        self._array = np.array([], dtype=self.DTYPE)
    
    def __getitem__(self, selector: Union[str, int, slice]):
        """ overload of getitem for convenience. basically the same as calling StructuredDeque._array[idx] but this avoids having to specify _array every time.

        Args
        ----
        - `selector`: can be either slice, index or fieldname
        """
        
        return self._array[selector]
       
    def __setitem__(self, selector: Union[str, int, slice], value: Any):
        """ overload of setitem for assigning values to the contents of _array. basically the same as calling StructuredDeque._array[idx] = [---] but this avoids havint to specify _array every time.
        
        Args
        ----
        - `selector`: can be either slice, index or fieldname
        """
        
        self._array[selector] = value

    def __repr__(self):
        """ overload of the repr method for displaying the object. standard fallback for str!. display() calls this first though, print() only as fallback. 
        """
        
        # get the names and shapes for all the columns
        names  = list(self._array.dtype.names)
        shapes = []
        for i in range(len(self._array.dtype)): # np dtype is not iterable!
            shapes.append(self._array.dtype[i].shape[0] if len(self._array.dtype[i].shape) > 0 else 1) 

        # repeat-expand names for each "subfield" that has a length of more than 1
        header_exp = [] # list of strings
        for _name, _shape in zip(names, shapes):
            if _shape > 1:
                for i in range(_shape):
                    header_exp += [f"{_name}[{i}]"]
            else:
                header_exp += [f"{_name}"]
        
        # collect all the data in one np array for convenience (conversion at once with astype is not allowed)
        concat_data = None
        for key in self._array.dtype.names:
            # extract a column (can be of arbitrary width, depending on field width)
            if len(self._array[key].shape) == 1:
                # "unsqueeze" because if the field has a width of one, the returned array is only one-dimensional
                column = self._array[key][:, None]
            else:
                column = self._array[key]
            # connect them all
            if concat_data is None:
                concat_data = column
            else:
                concat_data = np.concatenate([concat_data, column], axis=1)
                
        # determine cell width and helper function for truncating the content of each cell to display nicely
        max_width  = REPR_WIDTH
        w_initial  = int((max_width - len(header_exp) - 1)/len(header_exp))
        trunc_cell = lambda content, w: (content[:w-2] + " …") if (len(content) > w) else (content)
        
        # determine the leftover space that was lost because division has a rest
        total_string_len = (w_initial + 1) * len(header_exp) + 1 # account for line characters
        leftover         = max_width - total_string_len
        W                = [w_initial] * len(header_exp) # individual width for each column in a vector of widths
        W[0]            += leftover # modify the first column to take up the leftover space
        
        # build all the rows (use the list of widths to determine the with for each column)
        header_row    = "│" + "│".join([trunc_cell(f"{label:^{w}}", w) for w, label in zip(W, header_exp)]) + "│"
        sep_row       = "+" + "+".join("-" * w for w, _ in zip(W, header_exp))                              + "+"
        data_rows     = []
        for row in concat_data:
            formatted_row = "│" + "│".join([trunc_cell(f"{value:^{w}.6f}", w) for w, value in zip(W, row)]) + "│"
            data_rows.append(formatted_row)
        dtype_row = trunc_cell(f" StructuredDeque w/ fields: {self._array.dtype} ", max_width-4)
        
        # build the table (optionally add datatype information)
        table = [sep_row, header_row, sep_row] + data_rows + [sep_row] + [f"╰{dtype_row:┄^{max_width-2}}╯"]
        table = "\n".join(table)
        return table
        
    def __len__(self):
        """ overloading the len method for convenience. this avoids having to call len(structureddeque._array) every time and is just a shorthand.
        """
        
        return len(self._array)
      
    def __iter__(self) -> Iterator[Tuple]:
        """ overloading the iter method for convenience. avoids having to call iter(structureddeque._array) every time and is just a shorthand
        """
        
        return iter(self._array)  
    
    def append(self, newline: Tuple):
        """ basically works like to normal append of a collection's deque. appending until the maxlen is reached, then the oldest elements are pushed off. (appending means adding elements from the high-index side).

        Args:
        - `newline`: one new line of data to add. has to be a tuple which matches the original datatype!
        """
        
        # behaviour when the array is not at full length yet
        if len(self._array) < self.MAXLEN:
            newline     = np.array(newline, dtype=self.DTYPE)
            self._array = np.append(self._array, newline)
            return
        
        # behaviour when the array is at full length
        self._array[:-1] = self._array[1:] 
        self._array[-1]  = np.array(newline, dtype=self.DTYPE)
    
    def prepend(self, newline: Tuple):
        """ implements a "prepending" behaviour that works just like appending but from the other side. also pushes off elements when maxlen is reached! (prepending means adding elements from the low-index side).

        Args
        ----
        - `newline`: one new line of data to add. has to be a tuple which matches the original datatype!
        """
        
        # behaviour when the array is not at full length yet
        if len(self._array) < self.MAXLEN:
            newline     = np.array(newline, dtype=self.DTYPE)
            self._array = np.insert(self._array, 0, newline)
            return
        
        # behaviour when the array is at full length
        self._array[1:] = self._array[:-1]
        self._array[0]  = np.array(newline, dtype=self.DTYPE)
    
    def clear(self):
        """ just a convenience method for emtying the contents of the _array """
        
        self._array = np.array([], dtype=self.DTYPE)

class Log:
    """ minimal utility class for emulating the python loggin module but with ROS-friendly print statements """
    
    LEVEL = "DEBUG"

    def DEBUG(msg):
        if Log.LEVEL in ["DEBUG"]:
            print(f"\033[0m[DEBUG]: <HoleTracker> {msg}\033[0m")
        pass
        
    def ALERT(msg):
        if Log.LEVEL in ["DEBUG", "ALERT"]:
            print(f"\033[38;2;255;165;0m[ALERT]: <HoleTracker> {msg}\033[0m")
        pass
        
    def FATAL(msg):
        if Log.LEVEL in ["DEBUG", "ALERT", "FATAL"]:
            print(f"\033[38;2;255;0;0m[FATAL]: <HoleTracker> {msg}\033[0m")
        pass

class HoleTracker:
    def __init__(
        self, freq_visibility_check: float, freq_memory_check: float, freq_publish_estimate: float, 
        logging_level: str   = "DEBUG",   # from [DEBUG, ALERT, FATAL]
        tiebreak_method: str = "RANDOM",  # from [FIRST, RANDOM, KDE-Nx-BWx.x]
        update_method: str   = "REPLACE", # from [REPLACE, AVG-Nx, KDE-Nx-BWx.x]
        
        thr_ddetect: float   = 0.1,       # in m
        thr_imugaps: float   = 0.5,       # in s
        thr_inframe: float   = 2.0,       # in s
        thr_offrame: float   = 5.0,       # in s
        
        imu_hist_minlen: int = 100,       # should be enough to span delay in detection processing (~0.5s)
        imu_hist_maxlen: int = 2000,      # should be enough to span delay between two detections (~5s)
    ):
        """ main class for a configurable object tracker using imu data
                
        Args
        ----
        - `freq_visibility_check`: frequency at which the visibility check is run in outside loop
        - `freq_memory_check`: (not used) frequency at which the memory reset check is run in outside loop
        - `freq_publish_estimate`: (not used) frequency at which the estimate is published in outside loop
        
        Optional Parameters
        -------------------
        - `logging_level`: log message filter level from [DEBUG, ALERT, FATAL]
        - `tiebreak_method`: method for initializing the tracking process from [FIRST, RANDOM, KDE-Nx-BWx.X]
        - `update_method`: method for how the estimate is updated from detections FROM [REPLACE, AVG-Nx, KDE-Nx-BWx.X]
        - `thr_ddetect`: dist [m] threshold to judge whether a new detection is corresponding to the current target
        - `thr_imugaps`: time [s], after which memory is reset, when no new imu reading is received
        - `thr_inframe`: time [s], after which memory is reset, without new detection while estimate is visible
        - `thr_offrame`: time [s], after which memory is reset, without new detection while estimate is invis.
        - `imu_hist_minlen`: minimum enforced number of historical imu readings and estimates stored (should span the delay time that include everythign from the image being taken to the detections being received by the tracker!)
        - `imu_hist_maxlen`: number of historical imu readings stored (should span the time between two detections for propagating accumulated detection points!)
        """
        
        # sanitize inputs ------------------------------------------------------
        if not bool(re.compile(r"^(DEBUG|ALERT|FATAL)$").match(logging_level)):
            raise ValueError(f"please choose a valid logging level from [DEBUG, ALERT, FATAL]!")
        
        if any(e <= 0 for e in [freq_visibility_check, freq_memory_check, freq_publish_estimate]):
            raise ValueError(f"please choose all frequencies > 0!")
        if any(e <= 0 for e in [thr_imugaps, thr_inframe, thr_offrame]):
            raise ValueError(f"please choose time-thresholds > 0!")
        if thr_ddetect <= 0:
            raise ValueError(f"please choose detection threshold > 0!")
        
        if any(e <= 0 for e in [imu_hist_minlen, imu_hist_maxlen]):
            raise ValueError(f"please choose imu history lengths > 0!")
        if imu_hist_minlen >= imu_hist_maxlen:
            raise ValueError(f"please choose IMU_HIST_MINLEN < IMU_HIST_MAXLEN!")
        
        if not bool(re.compile(r"^(FIRST|RANDOM|KDE-N(?!0+\b)\d+-BW\d+\.\d+)$").match(tiebreak_method)):
            raise ValueError(f"please choose a valid tiebreak method from [FIRST, RANDOM, KDE-Nx-BWx.x]!")
        if not bool(re.compile(r"^(REPLACE|AVG-N(?!0+\b)\d+|KDE-N(?!0+\b)\d+-BW\d+\.\d+)$").match(update_method)):
            raise ValueError(f"please choose a valid update method from [REPLACE, AVG-Nx, KDE-Nx-BWx.x]")
        
        # initialize members ---------------------------------------------------
        Log.LEVEL = logging_level # set logging level for all print statements using Log.XXXXX
        
        # store frequencies for visibility check, memory reset check and estimate publishing
        self._FREQ_VISIBILITY_CHECK = freq_visibility_check
        self._FREQ_MEMORY_CHECK     = freq_memory_check     # not yet required
        self._FREQ_PUBLISH_ESTIMATE = freq_publish_estimate # not yet required
        
        self._TIEBREAK_METHOD = tiebreak_method.split("-")[0]
        self._TIEBREAK_N      = None # number of detections until initialization
        self._TIEBREAK_BW     = None # the bandwidth for KDE
        self._tiebreak_n_cnt  = None # temporary counter to keep track of how many detections have been aggregated
        self._tiebreak_cloud  = None # container for aggregating all the detection points
        self._tiebreak_old_ts = None # to keep track of which timestep the detections were propagated to
        if self._TIEBREAK_METHOD == "KDE":
            self._TIEBREAK_N     = int(tiebreak_method.split("-")[1][1:])
            self._TIEBREAK_BW    = float(tiebreak_method.split("-")[2][2:])
            self._tiebreak_n_cnt = 0
        
        self._UPDATE_METHOD  = update_method.split("-")[0]
        self._UPDATE_N       = None # number of detection points being kept in history and used for updating estimate
        self._UPDATE_BW      = None # bandwidth for KDE
        self._p_detection    = None
        if self._UPDATE_METHOD == "REPLACE":
            self._p_detection = StructuredDeque(
                maxlen   = 1,
                datatype = [("ts", np.float64, (1, )), ("p", np.float32, (3, ))]
            )
        if self._UPDATE_METHOD == "AVG":
            self._UPDATE_N    = int(update_method.split("-")[1][1:])
            self._p_detection = StructuredDeque(
                maxlen   = int(update_method.split("-")[1][1:]),
                datatype = [("ts", np.float64, (1, )), ("p", np.float32, (3, ))]
            )
        if self._UPDATE_METHOD == "KDE":
            self._UPDATE_N    = int(update_method.split("-")[1][1:])
            self._UPDATE_BW   = float(update_method.split("-")[2][2:])
            self._p_detection = StructuredDeque(
                maxlen   = int(update_method.split("-")[1][1:]),
                datatype = [("ts", np.float64, (1, )), ("p", np.float32, (3, ))]
            )
        
        self._THR_IMUGAPS = thr_imugaps # s
        self._THR_INFRAME = thr_inframe # s
        self._THR_OFFRAME = thr_offrame # s
        self._THR_DDETECT = thr_ddetect**2 # m, using **2 because its compared to dist**2 in this class
        
        self._IMU_HIST_MINLEN = imu_hist_minlen
        self._IMU_HIST_MAXLEN = imu_hist_maxlen
        
        self._imu_data = StructuredDeque(# imu deque
            maxlen   = int(imu_hist_maxlen), # imu history has to maximally span the interval of two detections!
            datatype = [("ts", np.float64, (1, )), ("twist", np.float32, (6, ))]
            )
        
        self._p_estimate  = StructuredDeque( # estimate deque
            maxlen   = int(imu_hist_minlen), # estimate (and imu) have to minimally span detection process delay (~0.5s)
            datatype = [("ts", np.float64, (1, )), ("p", np.float32, (3, )), ("vp", np.float32, (3, ))]
            )
        
        # other internal stores
        self._visibility_hist    = np.array([])
        self._flag_tracking      = False
        self._flag_new_detection = False
        
        Log.DEBUG(f"initialized tracker!")

    def __repr__(self):
        """ overload representation method to control how the object is printed when using print(object_instance). prints a nice summary of the tracker memory. """

        max_width = REPR_WIDTH
        
        def _format_string(width: int, name: str, value: Any, suffix: str=""):
            """ convenience for nicely formatting and padding info strings for the tracker repr method """
            
            suffix_str  = f" {suffix}" if suffix else "" # ensure suffix has a leading space if it's not empty
            base_str    = f"{name} {value}{suffix_str}" # base string without dots
            dots_needed = width - len(base_str) - 3  # calculate avail. space for dots (-2 brackets) (-1 added space)

            # construct the final formatted string with variable amounts of dots
            return f"[{name} {'┄' * dots_needed} {value}{suffix_str}]"
        
        output = (
            "\n" + f"╔{' TRACKER SUMMARY ':═^{max_width-2}}╗"                              + "\n" + "\n" + 
            
            _format_string(max_width, "FREQ_VISIBILITY_CHECK", self._FREQ_VISIBILITY_CHECK, "Hz") + "\n" + 
            _format_string(max_width, "FREQ_MEMORY_CHECK", self._FREQ_MEMORY_CHECK, "Hz")         + "\n" + 
            _format_string(max_width, "FREQ_PUBLISH_ESTIMATE", self._FREQ_PUBLISH_ESTIMATE, "Hz") + "\n" +
            
            _format_string(max_width, "LOGGING_LEVEL", Log.LEVEL)                                 + "\n" +  
            _format_string(max_width, "TIEBREAK_METHOD", self._TIEBREAK_METHOD)                   + "\n" + 
            _format_string(max_width, "TIEBREAK_N", self._TIEBREAK_N)                             + "\n" + 
            _format_string(max_width, "TIEBREAK_BW", self._TIEBREAK_BW)                           + "\n" + 
            _format_string(max_width, "UPDATE_METHOD", self._UPDATE_METHOD)                       + "\n" + 
            _format_string(max_width, "UPDATE_N", self._UPDATE_N)                                 + "\n" + 
            _format_string(max_width, "UPDATE_BW", self._UPDATE_BW)                               + "\n" + 
            
            _format_string(max_width, "THR_DDETECT", self._THR_DDETECT**0.5, "m")                 + "\n" +
            _format_string(max_width, "THR_IMUGAPS", self._THR_IMUGAPS, "s")                      + "\n" +
            _format_string(max_width, "THR_INFRAME", self._THR_INFRAME, "s")                      + "\n" +
            _format_string(max_width, "THR_OFFRAME", self._THR_OFFRAME, "s")                      + "\n" + 
            
            _format_string(max_width, "IMU_HIST_MINLEN", self._IMU_HIST_MINLEN)                   + "\n" + 
            _format_string(max_width, "IMU_HIST_MAXLEN", self._IMU_HIST_MAXLEN)                   + "\n" + 
            
            "\n" + f"╭{' IMU data ':┄^{max_width-2}}╮"    + "\n" + f"{self._imu_data}"            + "\n" + 
            "\n" + f"╭{' p_detection ':┄^{max_width-2}}╮" + "\n" + f"{self._p_detection}"         + "\n" + 
            "\n" + f"╭{' p_estimate ':┄^{max_width-2}}╮"  + "\n" + f"{self._p_estimate}"          + "\n" + 
            
            "\n" + f"╚{'═'*(max_width-2)}╝"                                                       + "\n"
        )

        return output
    
    def _kill_memory(self):
        """ resets the tracker memory (p_detection, p_estimate, visibility, tracking flag and new detection flag + tiebreak related temp stores), only the imu data history is preserved. """
        
        Log.DEBUG(f"killed memory!")
        
        self._p_detection.clear()
        self._p_estimate.clear()
        self._visibility_hist    = np.array([])
        self._flag_tracking      = False
        self._flag_new_detection = False # this is not strictly necessary here
        
        # just in case, reset initialization related containers. (is reset after successfull initialization as well)
        self._tiebreak_n_cnt  = 0
        self._tiebreak_cloud  = None
        self._tiebreak_old_ts = None
      
    def _add_imu_data(self, ts: float, data: List, method: str="append"):
        """ adds one imu sample to the tracker memory (appending to self.imu_data)

        Args
        ----
        - `t_imu`: timestamp (epoch) of the new imu sample
        - `data`: IMU measurement in the form [Vx, Vy, Vz, Wx, Wy, Wz] of the drone body as list
        - `method`: choose either appending or prepending [append, prepend]
        """
        
        # handle wrong usage of method
        if not isinstance(data, list):
            raise TypeError(f"trying to add imu_data but data input is not a list!")
        if self._imu_data.DTYPE[1][2][0] != len(data):
            raise TypeError(
                f"found wrong length in imu data to be added! got input length = {len(data)} but require length = {self._imu_data.DTYPE[1][2][0]}"
                )
        if method not in ["append", "prepend"]:
            raise ValueError(f"please choose a valid method for adding data from [append, prepend]! (got: {method})")

        # warn about adding samples in such a way that their timestamps do not increase monotonically 
        if len(self._imu_data) > 0:
            if (method == "append")  and (ts <= self._imu_data[-1]["ts"]):
                Log.FATAL(
                    f"function _add_imu_data (append) was trying to add data to the deque, but the new timestamp would break the criterium of the timestamps having to increase strictly monotinically. The tracker assumes that all timestamps in a deque increase monotonically, so this could cause problems in other methods! (got new ts: {ts} and the next relevant ts: {self._imu_data[-1]['ts'].squeeze()})"
                    )
                
            if (method == "prepend") and (ts >= self._imu_data[0]["ts"]):
                Log.FATAL(
                    f"function _add_imu_data (prepend) was trying to add data to the deque, but the new timestamp would break the criterium of the timestamps having to increase strictly monotinically. The tracker assumes that all timestamps in a deque increase monotonically, so this could cause problems in other methods! (got new ts: {ts} and the next relevant ts: {self._imu_data[0]['ts'].squeeze()})"
                    )
        
        # add sample
        if method == "append":
            self._imu_data.append((ts, data))        
        if method == "prepend":
            self._imu_data.prepend((ts, data))

    def _add_p_estimate(self, ts: float, data_p: List, data_vp: List, method: str="append"):
        """ adds the newest p_estimator to the deque (appending to self._p_estimate)

        Args
        ----
        - `ts`: timestamp (epoch) of when the new estimate was created. (delta to this ts will be used to make a pred.)
        - `data_p`: starting point for prediction in the form [X, Y, Z]
        - `data_vp`: current velocity of this starting point in drone coordinates in the form [Vpx, Vpy, Vpz]
        - `method`: choose either appending or prepending [append, prepend]
        """

        if not isinstance(data_p, list) or not isinstance(data_vp, list):
            raise TypeError(f"trying to add a p_prediction but data input is not a list!")
        if self._p_estimate.DTYPE[1][2][0] != len(data_p):
            raise TypeError(
                f"found wrong length in new estimate p data to be added! got input length = {len(data_p)} but require length = {self._p_estimate.DTYPE[1][2][0]}"
                )
        if self._p_estimate.DTYPE[2][2][0] != len(data_vp):
            raise TypeError(
                f"found wrong length in new estimate vp data to be added! got input length = {len(data_vp)} but require length = {self._p_estimate.DTYPE[2][2][0]}"
                )  
        if method not in ["append", "prepend"]:
            raise ValueError(f"please choose a valid method for adding data from [append, prepend]! (got: {method})")

        # warn about adding samples in such a way that their timestamps do not increase monotonically 
        if len(self._p_estimate) > 0:
            if (method == "append") and (ts <= self._p_estimate[-1]["ts"]):
                Log.FATAL(
                    f"function _add_p_estimate (append) was trying to add data to the deque, but the new timestamp would break the criterium of the timestamps having to increase strictly monotinically. The tracker assumes that all timestamps in a deque increase monotonically, so this could cause problems in other methods! (got new ts: {ts} and the next relevant ts: {self._p_estimate[-1]['ts'].squeeze()})"
                    )
            if (method == "prepend") and (ts >= self._p_estimate[0]["ts"]):
                Log.FATAL(
                    f"function _add_p_estimate (prepend) was trying to add data to the deque, but the new timestamp would break the criterium of the timestamps having to increase strictly monotinically. The tracker assumes that all timestamps in a deque increase monotonically, so this could cause problems in other methods! (got new ts: {ts} and the next relevant ts: {self._p_estimate[0]['ts'].squeeze()})"
                    )
        
        # add sample
        if method == "append":
            self._p_estimate.append((ts, data_p, data_vp))
        if method == "prepend":
            self._p_estimate.prepend((ts, data_p, data_vp))
  
    def _add_p_detection(self, ts: float, data: List, method: str="append"):
        """ adds the newest and confirmed detection to the deque (appending to self._p_detection). When multiple detections are stored (for update methods AVG and KDE) the previous points are also propagated to the new timestep.

        Args
        ----
        - `t_imu`: timestamp (epoch) of the new and confirmed detection
        - `data`: coordinates of newest detection point in the form [X, Y, Z] in drone coords (not image coords!)
        - `method`: choose either appending or prepending [append, prepend]
        """

        # handle wrong usage of method
        if not isinstance(data, list):
            raise TypeError(
                f"trying to add a p_detection but data input is not a list!"
                )
        if self._p_detection.DTYPE[1][2][0] != len(data):
            raise TypeError(
                f"found wrong length in new detection point data to be added! got input length = {len(data)} but require length = {self._p_detection.DTYPE[1][2][0]}"
                )
        if method not in ["append", "prepend"]:
            raise ValueError(
                f"please choose a valid method for adding p_detection from [append, prepend]! (got: {method})"
                )
        
        if self._UPDATE_METHOD == "REPLACE":
            # just append like normal, there will only be the newest point and deque of size 1
            pass
        
        if (self._UPDATE_METHOD == "AVG" or self._UPDATE_METHOD == "KDE") and len(self._p_detection) > 0:
            # first pull all the old detections forward up to the ts of the new detection
            
            ts_old   = self._p_detection["ts"][0].squeeze() # the timestamps should always all be the same anyways
            ts_new   = ts
            points   = copy.deepcopy(self._p_detection["p"])
            imu_data = copy.deepcopy(self._imu_data)
            
            if ts_old < imu_data[0]["ts"]:
                Log.FATAL(
                    f"when adding p_detection while trying to update the older stored detections, the imu history didn't reach all the way back to the old timestamp (od_ts = {ts_old}) (oldest imu sample = {imu_data[0]['ts']})! consider increasing the imu history maxlen."
                )
            if ts_old > ts_new:
                Log.FATAL(
                    f"when adding p_detection while trying to update the older storded detections, found the new detection points to have an older timestamp than the old detections (new ts = {ts_new}) (old detections ts = {ts_old}!"
                )

            # find the indices for the relevant window of imu data to propagate old detections forward
            sta_idx = np.clip(np.searchsorted(imu_data["ts"].squeeze(), ts_old) - 1, a_min=0, a_max=None)
            end_idx = np.clip(np.searchsorted(imu_data["ts"].squeeze(), ts_new) - 1, a_min=0, a_max=None)
            
            # extract the relevant imu steps and corresponding delta_ts
            relevant_imu      = imu_data["twist"][sta_idx:end_idx+1, :]
            relevant_ts       = imu_data["ts"]   [sta_idx:end_idx+1, :]
            relevant_ts[0, 0] = ts_old # the first inverval is only from old_ts to the next imu step!
            relevant_ts       = np.append(relevant_ts, np.array([[ts_new]]), axis=0) # last inverval only up to new_ts!
            delta_ts          = np.diff(relevant_ts, axis=0)
            
            # update the points through all these relevant timesteps (so that they are comparable to the new detection)
            points = points.T # transpose for batched updates with matrix multiplication
            for dt, twist in zip(delta_ts, relevant_imu):
                v      = twist[0:3]
                w      = twist[3:6]
                w_skew = np.array([
                    [    0, -w[2],  w[1]],
                    [ w[2],     0, -w[0]], 
                    [-w[1],  w[0],     0],  
                ])
                v_p    = -w_skew @ points - v[:, None]
                points = points + v_p * dt   
            points = points.T # to match input shape again
                
            self._p_detection["p"]  = points # replace the old detections with the newly updated ones
            self._p_detection["ts"] = ts_new # update their timestamps to what they have been updated to

        # finally add the new detection point
        if method == "append":
            self._p_detection.append((ts, data))
        if method == "prepend":
            self._p_detection.prepend((ts, data)) # doesn't make much sense in this context
        return
    
    def _update_estimate_from_estimate(self):
        """ This is only triggered exactly upon receiving new IMU data. For this function, assume that the new imu has already been stored in memory (by _add_imu method) and then this function is called afterwards. 
        
        Looks at the latest available estimate and evaluates it at the newest imu sample's timestamp. This obtained point then serves as the starting point for the new estimate! The new estimate is then valid from the new imu sample's timestamp onwards and uses the newly obtained camera twist to actually calculate a concrete estimate point when calling get_tracker_estimate.
        
        Should also handle gracefully being called when the IMU was not actually updated in the meantime! (although that shouldn't happen) """
        
        # handle wrong usage of method
        if self._flag_tracking is False:
            raise ValueError(
                f"trying to update estimate from estimate but currently no object is tracked!"
            )
        if len(self._p_estimate) == 0:
            raise ValueError(
                f"trying to update estimate from estimate but no previous estimate is available!"
            )
        if len(self._imu_data) < self._IMU_HIST_MINLEN:
            raise ValueError(
                f"trying to update estimate from estimate but full imu history missing (only got {len(self._imu_data)} / {self._IMU_HIST_MINLEN} measurements)"
            )

        # sanity check: is the update from estimate being called while a new detection would be available?
        if self._flag_new_detection is True:
            Log.ALERT(
                f"in _update_estimate_from_estimate, trying to update estimate from estimate but a new detection would be available to update from!"
            )
        # sanity check: was there actually a new imu measurement received since the creation of the last estimation?
        if np.round(self._p_estimate[-1]["ts"], decimals=6) == np.round(self._imu_data[-1]["ts"], decimals=6):
            Log.ALERT(
                f"in _update_estimate_from_estimate, trying to update estimate from estimate but no new imu reading was made since last estimate update! (skipping the update...)"
            )
            return
    
        # grab the newest imu measurement sample and its timestep
        new_imu_ts   = self._imu_data[-1]["ts"]
        new_imu_data = self._imu_data[-1]["twist"]
        
        # grab the newest estimate parameters (ts, p, vp)
        last_estimate_ts = self._p_estimate[-1]["ts"]
        last_estimate_p  = self._p_estimate[-1]["p"]
        last_estimate_vp = self._p_estimate[-1]["vp"]
                
        # evaluate the newest estimate at the timestamp of the newest imu reading (p_new = p + vp * delta_t)
        p_eval  = last_estimate_p + last_estimate_vp * (new_imu_ts - last_estimate_ts)
        
        # calculate the new velocity of the tracked point in camera frame (from camera twist: vp = - v_cam - w_cam x p)
        vp_eval = - new_imu_data[0:3] - np.cross(new_imu_data[3:6], p_eval)
        
        # update the current p_estimate 
        self._add_p_estimate(new_imu_ts, list(p_eval), list(vp_eval))

    def _update_estimate_from_detection(self):
        """ This is only triggered exactly upon receiving new IMU data AND when a new detection has been made (flag_new_detection = True). This function assumes that the new imu data AND new detection have already been saved in memory with a timestamp (by _add_p_detection and _add_imu_data).
        
        This can either totally initialize the hole tracking (when flag_tracking = False) or just update the currently tracked hole but now from a detection (or from multiple, depending on the update method) rather than just the last estimate (when flag_tracking = True)
        
        Basicall takes the latest detection (or multiple), then, depending on its timestamp and the delay it went through for processing, the measurement will be "brought to the newest timestep" by integrating it through all imu steps. """
        
        # handle wrong usage of method
        if len(self._p_detection) == 0:
            raise ValueError(
                f"trying to update estimate from detection but no detection is available!"
            )        
        if len(self._imu_data) < self._IMU_HIST_MINLEN:
            raise ValueError(
                f"trying to update estimate from detection but full imu history missing (only got {len(self._imu_data)} / {self._IMU_HIST_MINLEN} measurements)"
            )
        if self._flag_new_detection is False:
            raise ValueError(
                f"trying to update estimate from detection but new_detection flag is False!"
            )

        # sanity check: does the imu history reach all the way back to the detection time?   
        if self._p_detection[-1]["ts"] < self._imu_data[0]["ts"]:
            Log.FATAL(
                f"in _update_estimate_from_detection, oldest saved imu time (={self._imu_data[0]['ts'].squeeze()}) does not reach as far back as the new detection time (={self._p_detection[-1]['ts'].squeeze()}) (with delta_t: {self._imu_data[0]['ts'].squeeze() - self._p_detection[-1]['ts'].squeeze()})! consider increasing the imu history minlen."
            )
        # sanity check: is the detection newer than the newest imu measurement (not a problem but should not happen)
        if self._p_detection[-1]["ts"] > self._imu_data[-1]["ts"]:
            Log.ALERT(
                f"in _update_estimate_from_detection, even the newest imu sample is older than the new detection! unproblematic if delta_t is small, but should not ocurr. newest saved imu t is: {self._imu_data[-1]['ts'].squeeze()} and new detection t is: {self._p_detection[-1]['ts'].squeeze()} with delta_t: {self._p_detection[-1]['ts'].squeeze() - self._imu_data[-1]['ts'].squeeze()}. This can happen when no imu readings were made in a while, when working with simulated data or when the detection procesing delay is very short."
            )

        if self._UPDATE_METHOD == "REPLACE":
            # just take the one point that is stored in the detection an propagate it forward in time
            detect_p  = self._p_detection[-1]["p"]
            detect_ts = self._p_detection[-1]["ts"]
        
        if self._UPDATE_METHOD == "AVG":
            # take all the detection points and compute their center of mass. then propagate this one point forward
            detect_p  = np.mean(self._p_detection["p"], axis=0)
            detect_ts = self._p_detection[-1]["ts"] # all ts should be the same anyways
            
        if self._UPDATE_METHOD == "KDE":
            # apply kde to the detection points and get the max density point (or wgh avg). then pull this point forward
            densities, _ = kde_sklearn(self._p_detection["p"], bandwidth=self._UPDATE_BW) # highest density point
            softmax_temp = 10 # controls the "sharpness" of softmax
            density_norm = softmax((densities/softmax_temp)[:, None], axis=0) # normalize density 
            detect_p     = np.sum(self._p_detection["p"] * density_norm, axis=0) # take norm-density weighted C.o.M.
            detect_ts    = self._p_detection[-1]["ts"] # all ts should be the same anyways
        
        # independent of the update method, each spit out one point representing the detections + detection timestamp
        detect_ts = detect_ts # just for overview completeness
        detect_p  = copy.deepcopy(detect_p )
        imu_data  = copy.deepcopy(self._imu_data)
        
        # find the indices for the relevant window of imu data to propagate old detections forward
        sta_idx = np.clip(np.searchsorted(imu_data["ts"].squeeze(), detect_ts.squeeze()) - 1, a_min=0, a_max=None)
        end_idx = len(imu_data) - 1 # always update the estimate up to the latest imu measurement
        if sta_idx == end_idx: 
            sta_idx = sta_idx - 1 # in case the detection is actually newer than the newest imu, just safeguard

        # extract the relevant imu steps and corresponding delta_ts
        relevant_imu      = imu_data["twist"][sta_idx:end_idx,   :] # here, last one is not needed, only later for estim
        relevant_ts       = imu_data["ts"]   [sta_idx:end_idx+1, :]
        relevant_ts[0, 0] = detect_ts # the first inverval is only from old_ts to the next imu step!
        delta_ts          = np.diff(relevant_ts, axis=0)
        
        # update the slightly old detection through all these relevant timesteps (up to the latest imu measurement)
        detect_p_upd = detect_p[:, None]
        for dt, twist in zip(delta_ts, relevant_imu):
            v      = twist[0:3]
            w      = twist[3:6]
            w_skew = np.array([
                [    0, -w[2],  w[1]],
                [ w[2],     0, -w[0]], 
                [-w[1],  w[0],     0],  
            ])
            v_p          = -w_skew @ detect_p_upd - v[:, None]
            detect_p_upd = detect_p_upd + v_p * dt  
        detect_p_upd = detect_p_upd.squeeze()

        # calculate the new velocity of the detected point in camera frame (from camera twist: vp = - v_cam - w_cam x p)
        detect_vp = - imu_data["twist"][-1][0:3] - np.cross(imu_data["twist"][-1][3:6], detect_p_upd)
        
        # finally create the new p_estimate with the newest imu timestep and the freshly updated detection point
        self._add_p_estimate(imu_data["ts"][-1].squeeze(), list(detect_p_upd), list(detect_vp))
   
    def _initialize_estimate_from_detection(self):
        """ just for documentation / clear naming purposes. initializing from detection works exactly the same way as updating from detection. The only addition for initialization is the one-time "backwards population" of all the historical p_estimates. this is needed for new detection logic to be prepared for large delays in the detection processing (this way it always has a full history of p_estimates). """
        
        # handle wrong usage of method
        if len(self._p_estimate) != 0:
            raise ValueError(f"trying to initialize estimate from detection but p_estimate is not empty!")
        
        self._update_estimate_from_detection()  
    
        # additionally, for the initialization "populate all the estimates backwards" (p_old = p_new - vp * delta_t)
        for i in range(2, min(len(self._imu_data), self._IMU_HIST_MINLEN) + 1):
            # inverting [p_k+1 = p_k + vp_k*Δt], where [vp_k = -v_cam_k - w_cam_k x p_k] to find the old starting point 
            # => p_k = inv(I - Δt*Ω_k) @ (p_k+1 + Δt*v_cam_k), where Ω = "cross-product-matrix" for w_cam_k
            v_cam_older = self._imu_data[-i]["twist"][0:3]
            w_cam_older = self._imu_data[-i]["twist"][3:6]
            Omega_older = np.array([
                [              0, -w_cam_older[2],  w_cam_older[1]],
                [ w_cam_older[2],               0, -w_cam_older[0]],
                [-w_cam_older[1],  w_cam_older[0],               0],
            ], dtype=np.float32)
            delta_t     = self._p_estimate[-i + 1]["ts"] - self._imu_data[-i]["ts"]
            p_newer     = self._p_estimate[-i + 1]["p"]
            p_older     = np.linalg.inv(np.eye(3, 3) - delta_t*Omega_older) @ (p_newer + delta_t*v_cam_older)
            vp_older    = - v_cam_older - np.cross(w_cam_older, p_older)
            
            # add the newly found old estimate to the list (but prepending to deque here!)
            self._add_p_estimate(self._imu_data[-i]["ts"], list(p_older), list(vp_older), method="prepend")
        
    def _detection_tiebreak(self, ts: float, detections: np.ndarray):
        """ method for choosing one of the multiple detections when initializing the tracking target from a new detection. Assumes that the KDE params are stored as instance members.

        Args
        ----
        - `detections`: coordinates of all detection points in the form [n_detects, 3] => n x [X, Y, Z] in drone frame
        - `method`: method for drawing a point from a multi detection for initializing [FIRST, RANDOM, KDE]
        """
        
        if self._TIEBREAK_METHOD == "FIRST":
            return detections[0, :]
        
        if self._TIEBREAK_METHOD == "RANDOM":
            return detections[np.random.randint(low=0, high=detections.shape[0]), :]
        
        if self._TIEBREAK_METHOD == "KDE":
            
            # just "initialize the tiebreaking" --------------------------------
            if self._tiebreak_cloud is None: 
                self._tiebreak_old_ts  = ts
                self._tiebreak_cloud   = detections # np.ndarray [n, 3]
                self._tiebreak_n_cnt  += 1 #just a counter to keep track of how many detections were aggregated
                return None
            
            # if already initialized -------------------------------------------
            # 1) update all the previous points to the current timestep (ts)
            # 2) add the new detections to the updated cloud

            # find the relevant imu steps to update the old detections to the new ts
            imu_data = copy.deepcopy(self._imu_data)
            
            if self._tiebreak_old_ts < imu_data[0]["ts"]:
                Log.FATAL(
                    f"when tiebreaking and accumulating detection points, the oldest stored imu sample did not reach as far back as the timestamp of the old detection points (old detections ts = {self._tiebreak_old_ts}) (oldest imu ts = {imu_data[0]['ts']})! consider increasing the imu history maxlen."
                )
            
            sta_idx  = np.clip(np.searchsorted(imu_data["ts"].squeeze(), self._tiebreak_old_ts)-1, a_min=0, a_max=None)
            end_idx  = np.clip(np.searchsorted(imu_data["ts"].squeeze(), ts)-1,                   a_min=0, a_max=None)
            
            # extract the relevant imu matrix and compute the corresponding delta_ts 
            relevant_imu      = imu_data["twist"][sta_idx:end_idx+1, :]
            relevant_ts       = imu_data["ts"]   [sta_idx:end_idx+1, :]
            relevant_ts[0, 0] = self._tiebreak_old_ts # the first inverval is only from old_ts to the next imu step!
            relevant_ts       = np.append(relevant_ts, np.array([[ts]]), axis=0) # last inverval is only up to new_ts!
            delta_ts          = np.diff(relevant_ts, axis=0)
            
            # update the points through all these relevant timesteps (so that they are comparable to the new detections)
            points = self._tiebreak_cloud.T
            for dt, twist in zip(delta_ts, relevant_imu):
                v      = twist[0:3]
                w      = twist[3:6]
                w_skew = np.array([
                    [    0, -w[2],  w[1]],
                    [ w[2],     0, -w[0]], 
                    [-w[1],  w[0],     0],  
                ]) 
                v_p    = -w_skew @ points - v[:, None]
                points = points + v_p * dt
            
            # add the new points to the updated ones
            self._tiebreak_cloud   = np.concatenate([points.T, detections])
            self._tiebreak_old_ts  = ts
            self._tiebreak_n_cnt  += 1
            
            Log.DEBUG(f"tiebreak step: {self._tiebreak_n_cnt}/{self._TIEBREAK_N} completed")
            
            # do KDE when enough points were collected -------------------------
            if self._tiebreak_n_cnt >= self._TIEBREAK_N:
                _, highest_density_p = kde_sklearn(self._tiebreak_cloud, self._TIEBREAK_BW)
                # reset memory for the next initialization
                self._tiebreak_n_cnt  = 0
                self._tiebreak_old_ts = None
                self._tiebreak_cloud  = None
                # finally return the newfound detection (ONE point)
                return highest_density_p

            return None # return None when not ready
            
    def do_new_detection_logic(self, ts: float, detections: np.ndarray):
        """ this method contains all the logic-decisions to process a new collection of detections. three main outcomes of this process (given that at least one point is passed to this function):
        
        - if [✘ currently tracking | ? the/any new detection plausible]
            ➼ tiebreak detections, ➼ add one detection to p_detection, ➼ flag_new_detection = True
        - if [✔ currently tracking | ✘ the/any new detection plausible] 
            ➼ no change, discard detections
        - if [✔ currently tracking | ✔ the/any new detection plausible] 
            ➼ add the best detection to p_detection, flag_new_detection = True, ➼ reset visibility to np.array[]  
        - except if the imu history isn't full yet, then this function is skipped
        
        Args
        ----
        - `ts`: timestamp (epoch) of when the new detections were made. should be timestamp of the image to be precise
        - `detections`: coordinates of all detection points in the form [n_detects, 3] => n x [X, Y, Z] in drone frame
        """
        
        # handle wrong usage of method
        if not isinstance(ts, float):
            raise TypeError(
                f"ts has to be a float!"
            )
        if not isinstance(detections, np.ndarray):
            raise TypeError(f"new detections input has to be a numpy array!")
        if len(detections.shape) != 2:
            raise TypeError(f"new detections input has to be a 2-dimensional array! (got {detections.shape})")
        if detections.shape[1] != 3:
            raise TypeError(
                f"new detections input has to be a 2-dimensional array of dimensions (n, 3)! "
                f" (got {detections.shape})"
                )
        if detections.shape[0] == 0:
            raise ValueError(f"new detections input is empty! (has to contain at least one detection point)")
        
        # skip detection logic if imu history is not yet full
        if len(self._imu_data) < self._IMU_HIST_MINLEN:
            Log.DEBUG(f"processing new detection: skipping because imu history is not yet full!")
            return
        
        # if NO HOLE IS TRACKED at the moment, tiebreak the detections and store one new detection point
        if self._flag_tracking is False:
            Log.DEBUG(f"tiebreaking the new detection! (might take mulitple do_detections for KDE method)")
            
            new_p = self._detection_tiebreak(ts, detections)
            if new_p is None:
                return # for the kde method. Only returns a tiebroken point after a few new detections
            
            Log.DEBUG(f"got a tiebreak result as an initial detection!")
            
            if self._UPDATE_METHOD == "REPLACE":
                self._add_p_detection(ts, list(new_p))
            
            if self._UPDATE_METHOD == "AVG" or self._UPDATE_METHOD == "KDE":
                # populate the detection store with x copies of the initial hole to give it a robust start
                for _ in range(self._UPDATE_N): self._add_p_detection(ts, list(new_p)) 
            
            self._flag_new_detection = True
            return
        
        # if A HOLE IS TRACKED at the moment, compare the detections to the current target. 
        
        # sanity check: does the estimate history reach all the way back to the new detection time?   
        if ts < self._p_estimate[0]["ts"]:
            Log.FATAL(
                f"in do_new_detection_logic oldest saved p_estimate does not reach as far back as the new detection time! not a problem if delta_t is small. oldest saved estimate time is: {self._p_estimate[0]['ts'].squeeze()} and new detection time is: {ts} with delta_t: {self._p_estimate[0]['ts'].squeeze() - ts}. consider increasing the imu history minlen (controls p_estimate minlen)"
            )

        # find the historical p_estimate that was "valid" during the time of ts_detection or take the most recent one
        idx = np.clip(np.searchsorted(self._p_estimate["ts"].squeeze(), ts) - 1, a_min=0, a_max=None)

        # evaluate that (historical) estimate at ts_new_detection
        p_eval = self._p_estimate[idx]["p"] + self._p_estimate[idx]["vp"] * (ts - self._p_estimate[idx]["ts"])
        
        # the new detections can now be compared to this p_eval
        distances = np.sum((detections - p_eval)**2, axis=1) # squared distances, for comparison it doesn't matter
        idx_best  = np.argmin(distances)
        
        # if even the best detection is NOT close enough to the estimate, discard detections
        if distances[idx_best] >= self._THR_DDETECT:
            Log.DEBUG(
                f"discarding detections because none is close enough to p_estimate! (closest was: {distances[idx_best]**0.5:.3f}m)"
                )
            return
        
        # the best new detection IS CLOSE ENOUGH to the estimate, take it as a new detection
        self._add_p_detection(ts, list(detections[idx_best]))
        self._flag_new_detection = True
        self._visibility_hist    = np.array([]) # reset visibility history
        Log.DEBUG(
            f"found a good detection! (closest was: {distances[idx_best]**0.5:.3f}m). also resetting visibility history!"
            )
        return
      
    def do_new_imu_logic(self, ts: float, new_imu: np.ndarray):
        """ this method contains all the logic-decision to process a new imu measurement reading. Four outcomes are possible:
        
        - if [✔ full IMU | ✔ tracking | ✔ new detection]
            ➼ save imu, ➼ update estimate from detection, ➼ flag_new_detection = False  
        - if [✔ full IMU | ✘ tracking | ✔ new detection] 
            ➼ save imu, ➼ initialize estimate from detection, ➼ flag_new_detection = False, flag_tracking = True
        - if [✔ full IMU | ✔ tracking | ✘ new detection]
            ➼ save imu, ➼ update estimate from estimate
        - else
            ➼ save imu, ➼ no change
            
        Args
        ----
        - `ts`: timestamp (epoch) of when the new imu measurement was made
        - `new_imu`: new drone twist in the form of a (1, 6) numpy array [[Vx, Vy, Vz, Wx, Wy, Wz]]
        """
        
        # handle wrong usage of method
        if not isinstance(ts, float):
            raise TypeError(
                f"ts has to be a float!"
            )
        if not isinstance(new_imu, np.ndarray):
            raise TypeError(f"new imu data has to be a numpy array!")
        if len(new_imu.shape) != 2:
            raise TypeError(f"new imu data input has to be a 2-dimensional array! (got {new_imu.shape})")
        if (new_imu.shape[1] != 6) or (new_imu.shape[0] > 1):
            raise TypeError(
                f"new imu data input has to be a 2-dimensional array of dimensions (1, 6)! "
                f" (got {new_imu.shape})"
                )
        if new_imu.shape[0] == 0:
            raise ValueError(f"new imu data input is empty! (has to contain exactly one row of data)")

        # choose action
        full_imu = len(self._imu_data) >= self._IMU_HIST_MINLEN
        
        if (full_imu) and (self._flag_tracking is True ) and (self._flag_new_detection is True ):
            self._add_imu_data(ts, list(new_imu.squeeze()))
            self._update_estimate_from_detection()
            self._flag_new_detection = False
            Log.DEBUG(f"processing new imu: updating estimate from detection!")
            return
        
        if (full_imu) and (self._flag_tracking is False) and (self._flag_new_detection is True ):
            self._add_imu_data(ts, list(new_imu.squeeze()))
            self._initialize_estimate_from_detection()
            self._flag_new_detection = False
            self._flag_tracking      = True
            Log.DEBUG(f"processing new imu: initializing estimate from detection!")
            return
        
        if (full_imu) and (self._flag_tracking is True ) and (self._flag_new_detection is False):
            self._add_imu_data(ts, list(new_imu.squeeze()))
            self._update_estimate_from_estimate()
            Log.DEBUG(f"processing new imu: updating estimate from estimate!")
            return
        
        self._add_imu_data(ts, list(new_imu.squeeze()))
        Log.DEBUG(f"processing new imu: doing nothing (only adding imu to memory)")
        # do nothing
        return
     
    def do_memory_check(self, ts: float):
        """ this function serves to check whether the ongoing target should be dropped or tracking should continue. three criteria for dropping the target (this is done by killing the memory):
        
        - not receiving an imu measurement for a certain time                                   => `THRESH_IMUGAPS [s]`
        - not getting new detection of the target for a certain time while it should be visible => `THRESH_INFRAME [s]`
        - not getting new detection of the target for a certain time while it is invisible      => `THRESH_OFFRAME [s]`
        
        Additionally this function raises an error, when a certain "impossible" / disallowed combination of memory elements is detected!
        
        Args
        ----
        - `ts`: timestamp (epoch) of when the check is run
        """
        
        # handle wrong usage of method
        if not isinstance(ts, float):
            raise TypeError(
                f"ts has to be a float!"
            )
        
        # handle disallowed memory configuration
        if (self._flag_tracking is True) and (len(self._p_estimate) < self._p_estimate.MAXLEN):
            Log.ALERT(
                f"memory check detected disallowed configuration: flag_tracking is True but p_estimate is not full! (got {len(self._p_estimate)} / {self._p_estimate.MAXLEN} estimates) (resetting memory...)" 
            )
            self._kill_memory()
            
        if (self._flag_tracking is True) and (len(self._imu_data) < self._IMU_HIST_MINLEN):
            Log.ALERT(
                f"memory check detected disallowed configuration: flag_tracking is True but imu_data is not full! (got {len(self._imu_data)} / {self._IMU_HIST_MINLEN} estimates) (resetting memory...)"  
            )
            self._kill_memory()
            
        if (self._flag_tracking is True) and (len(self._p_detection) == 0):
            Log.ALERT(
                f"memory check detected disallowed configuration: flag_tracking is True but p_detection is empty! (resetting memory...)"        
            ) 
            self._kill_memory()
            
        if (self._flag_tracking is False) and (len(self._p_estimate) != 0):   
            Log.ALERT(
                f"memory check detected disallowed configuration: flag_tracking is False but p_estimate is not empty! (got p_estimate len: {len(self._p_estimate)}) (resetting memory...)"  
            )   
            self._kill_memory()
            
        if (self._flag_tracking is False) and (len(self._visibility_hist) != 0):
            Log.ALERT(
                f"memory check detected disallowed configuration: flag_tracking is False but visibility history is not empty! (got visibity history len: {len(self._visibility_hist)}) (resetting memory...)"
            )
            self._kill_memory()
   
        # nothing to be reset when not tracking anything, skip the check
        if self._flag_tracking is False:
            return
        
        # if target is being tracked, do all three checks
        sum_inframe = np.sum(    self._visibility_hist) * (1/self._FREQ_VISIBILITY_CHECK) # time in frame, w\o detection
        sum_offrame = np.sum(1 - self._visibility_hist) * (1/self._FREQ_VISIBILITY_CHECK) # time off frame w\o detection
        delta_t_imu = ts - self._imu_data[-1]["ts"].squeeze()
  
        if sum_inframe >= self._THR_INFRAME:
            self._kill_memory()
            Log.DEBUG(f"killing memory because target was not detected for too long and is in frame!")
        if sum_offrame >= self._THR_OFFRAME:
            self._kill_memory()
            Log.DEBUG(f"killing memory because target was not detected for too long and is out of frame!")
        if delta_t_imu >= self._THR_IMUGAPS:
            self._kill_memory()
            Log.DEBUG(f"killing memory because no new imu data was received!")
        return
    
    def do_inframe_check(self, ts: float, estim2img: Callable[[np.ndarray], Tuple[np.ndarray, float]], img_res: tuple):
        """ Input to estim2img has to be a np.ndarray of shape (3,) (not homogenous coordinates). The output should be an np.ndarray of shape (2,) with img plane coordinates [u, v] and a float for the Z coordinate in camera frame.

        Args
        ----
        - `ts`: timestamp (epoch) of when the check is run
        - `estim2img`: function that transforms estimate (imu coords) to img plane [u, v] and returns Z in cam coords
        - `img_res`: tuple of image size of the form (img_w, img_h)
        """
        
        # handle wrong usage of method
        if not isinstance(ts, float):
            raise TypeError(
                f"ts has to be a float!"
            )
        if not callable(estim2img):
            raise TypeError("for inframe check the estim2img has to be a function (callable)!")
        if not isinstance(img_res, tuple):
            raise TypeError("for inframe check, the img_res has to be a tuple!")
        if len(img_res) != 2:
            raise TypeError(f"for inframe check the img_res tuple has to be of length 2! (got length {len(img_res)})")
        if (self._flag_tracking is True) and (len(self._p_estimate) == 0):
            raise ValueError("for inframe check flag_tracking is true but p_estimate is empty!")
        
        # skip the check if not tracking a target
        if self._flag_tracking is False:
            return
        
        img_w, img_h = img_res[0], img_res[1]
        
        # if the currently tracked target is available, evaluate it location @ time = ts
        p_eval   = self._p_estimate[-1]["p"] + self._p_estimate[-1]["vp"] * (ts - self._p_estimate[-1]["ts"])
        # transform the point to image plane coordinates using the function given as argument
        p_img, Z = estim2img(p_eval)
        
        if not isinstance(p_img, np.ndarray):
            raise TypeError(
                f"for inframe check estim2img returned not and ndarray for p in image plane! (got {type(p_img)})"
                )
        if not p_img.shape == (2,):
            raise TypeError(
                f"for inframe check estim2img returned the wrong shape for p in image plane! "
                f"(got {p_img.shape}) but need (2,)"
                )
        if not isinstance(Z, float):
            raise TypeError(f"for inframe check estim2img returned the wrong type for z! (got {type(Z)} need float)")
        
        in_image   = (p_img[0] >= 0) and (p_img[1] >= 0) and (p_img[0] <= img_w) and (p_img[1] <= img_h)
        positive_Z = Z > 0
        
        # append the resulting check result to the visibility history
        if in_image and positive_Z:
            self._visibility_hist = np.append(self._visibility_hist, 1)
            Log.DEBUG(f"doing inframe check: target was in frame!")
        else: 
            self._visibility_hist = np.append(self._visibility_hist, 0)
            Log.DEBUG(f"doing inframe check: target was out of frame!")
        return

    def get_tracker_estimate(self, ts: float) -> Union[np.ndarray, None]:
        """ this function returns the current best estimate evaluated a desired timestep ts.
        
        - when a target is tracked, returns a point in drone coordinates, np.ndarray of shape (1, 3) [X, Y, Z]
        - when no target is trached, returns None

        Args
        ----
        - `ts`: timestamp (epoch) of when to evaluate the current best estimate
        """
        
        # handle wrong usage of method
        if not isinstance(ts, float):
            raise TypeError(
                f"ts has to be a float!"
            )
        if (self._flag_tracking is True) and (len(self._p_estimate) == 0):
            raise ValueError(f"trying to return tracker estimate but flag_tracking is true and p_estimate is empty!")
        
        # sanity check
        if self._flag_tracking is True:
            if ts < self._p_estimate[-1]["ts"] - 0.1: # -0.1 because this otherwise this clutters the logging
                Log.ALERT(
                    f"in get_tracker_estimate returning tracker estimate but evaluation time is earlier than p_estimate creation ts! not a problem for if delta_t is small. evaluation time is: {ts} p_estimate creation time is: {self._p_estimate[-1]['ts'].squeeze()} delta_t is: {self._p_estimate[-1]['ts'].squeeze() - ts}."
                )
        
        # return estimate when tracking or None when not tracking        
        if self._flag_tracking is True:
            Log.DEBUG(f"calculating and returning the newest estimate")
            return (self._p_estimate[-1]["p"] + self._p_estimate[-1]["vp"] * (ts - self._p_estimate[-1]["ts"]))[None, :]
        else:
            Log.DEBUG(f"not tracking, returning None as estimate")
            return None

