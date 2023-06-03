import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostError
from typing import Iterable


class ConformalMultiQuantile(CatBoostRegressor):
    
    def __init__(self, quantiles:Iterable[float], *args, **kwargs):

        """
        Initialize a ConformalMultiQuantile object.

        Parameters
        ----------
        quantiles : Iterable[float]
            The list of quantiles to use in multi-quantile regression.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        kwargs['loss_function'] = self.create_loss_function_str(quantiles)
        super().__init__(*args, **kwargs)
        self.quantiles = quantiles
        self.calibration_adjustments = None
           
        
    @staticmethod
    def create_loss_function_str(quantiles:Iterable[float]):

        """
        Format the quantiles as a string for Catboost

        Paramters
        ---------
        quantiles : Union[float, List[float]]
            A float or list of float quantiles
        
        Returns
        -------
        The loss function definition for multi-quantile regression
        """

        quantile_str = str(quantiles).replace('[','').replace(']','')

        return f'MultiQuantile:alpha={quantile_str}'
                    
    def calibrate(self, x_cal, y_cal):

        """
        Calibrate the multi-quantile model

        Paramters
        ---------
        x_cal : ndarray
            Calibration inputs
        y_cal : ndarray
            Calibration target
        """

        # Ensure the model is fitted
        if not self.is_fitted():

            raise CatBoostError('There is no trained model to use calibrate(). Use fit() to train model. Then use this method.')
        
        # Make predictions on the calibration set
        uncalibrated_preds = self.predict(x_cal)

        # Compute the difference between the uncalibrated predicted quantiles and the target
        conformity_scores = uncalibrated_preds - np.array(y_cal).reshape(-1, 1)
        
        # Store the 1-q quantile of the conformity scores
        self.calibration_adjustments = \
            np.array([np.quantile(conformity_scores[:,i], 1-q) for i,q in enumerate(self.quantiles)])
        
    def predict(self, data, prediction_type=None, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):

        """
        Predict using the trained model.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            Data to make predictions on
        prediction_type : str, optional
            Type of prediction result, by default None
        ntree_start : int, optional
            Number of trees to start prediction from, by default 0
        ntree_end : int, optional
            Number of trees to end prediction at, by default 0
        thread_count : int, optional
            Number of parallel threads to use, by default -1
        verbose : bool or int, optional
            Verbosity, by default None
        task_type : str, optional
            Type of task, by default "CPU"

        Returns
        -------
        numpy.ndarray
            The predicted values for the input data.
        """
        
        preds = super().predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)

        # Adjust the predicted quantiles according to the quantiles of the
        # conformity scores
        if self.calibration_adjustments is not None:

            preds = preds - self.calibration_adjustments

        return preds