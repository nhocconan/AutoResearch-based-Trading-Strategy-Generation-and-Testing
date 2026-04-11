#!/usr/bin/env python3
"""
6h_1d_alligator_liner_breakout
Strategy: 6s Williams Alligator + 1d linear regression breakout with volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines Williams Alligator (Jaw/Teeth/Lips) from 6h for trend detection with 1d linear regression channel breakouts for entry timing. Uses volume > 1.5x 20-period average for confirmation. Designed to work in both bull (trend following with Alligator) and bear (mean reversion at extremes) by only taking breakouts in the direction of the Alligator alignment. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_alligator_liner_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 6h Williams Alligator (Smoothed SMAs)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        smma_vals = np.full_like(arr, np.nan, dtype=np.float64)
        smma_vals[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set shifted values to nan
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe (already 6h, so no alignment needed)
    jaw_6h = jaw
    teeth_6h = teeth
    lips_6h = lips
    
    # 1d Linear Regression Channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def linear_regression_channel(arr, period=30):
        """Calculate linear regression slope and intercept"""
        if len(arr) < period:
            return np.full_like(arr, np.nan), np.full_like(arr, np.nan)
        slope = np.full_like(arr, np.nan)
        intercept = np.full_like(arr, np.nan)
        
        for i in range(period-1, len(arr)):
            y = arr[i-period+1:i+1]
            x = np.arange(len(y))
            if np.all(np.isnan(y)):
                continue
            # Only use non-nan values
            valid = ~np.isnan(y)
            if np.sum(valid) < 2:
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            if len(x_valid) < 2:
                continue
            # Linear regression
            A = np.vstack([x_valid, np.ones(len(x_valid))]).T
            m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
            slope[i] = m
            intercept[i] = c
        return slope, intercept
    
    slope, intercept = linear_regression_channel(close_1d, 30)
    
    # Calculate upper and lower bands (1 standard deviation)
    def calculate_std_dev(arr, period=30):
        std_dev = np.full_like(arr, np.nan)
        for i in range(period-1, len(arr)):
            y = arr[i-period+1:i+1]
            valid = ~np.isnan(y)
            if np.sum(valid) < 2:
                continue
            y_valid = y[valid]
            x_valid = np.arange(len(y_valid))
            if len(x_valid) < 2:
                continue
            # Predict values
            y_pred = intercept[i] + slope[i] * x_valid
            # Standard deviation of residuals
            residuals = y_valid - y_pred
            std_dev[i] = np.std(residuals) if len(residuals) > 0 else np.nan
        return std_dev
    
    std_dev = calculate_std_dev(close_1d, 30)
    upper_band = intercept + slope * (29) + std_dev  # end of period
    lower_band = intercept + slope * (29) - std_dev
    
    # Align to 6h timeframe
    upper_band_6h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_6h = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(upper_band_6h[i]) or np.isnan(lower_band_6h[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Alligator alignment: Mouth open (trending) or closed (ranging)
        # Mouth open: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        # Mouth closed: all intertwined
        lips_above_teeth = lips_6h[i] > teeth_6h[i]
        teeth_above_jaw = teeth_6h[i] > jaw_6h[i]
        lips_below_teeth = lips_6h[i] < teeth_6h[i]
        teeth_below_jaw = teeth_6h[i] < jaw_6h[i]
        
        uptrend_alligator = lips_above_teeth and teeth_above_jaw
        downtrend_alligator = lips_below_teeth and teeth_below_jaw
        
        # Breakout conditions
        breakout_up = price_close > upper_band_6h[i]
        breakout_down = price_close < lower_band_6h[i]
        
        # Volume confirmation
        vol_ok = vol_confirmed[i]
        
        # Only trade in direction of Alligator alignment
        long_signal = breakout_up and vol_ok and uptrend_alligator
        short_signal = breakout_down and vol_ok and downtrend_alligator
        
        # Exit when price crosses the Alligator lines (Lips) or opposite band
        exit_long = position == 1 and (price_close < lips_6h[i] or price_close < lower_band_6h[i])
        exit_short = position == -1 and (price_close > lips_6h[i] or price_close > upper_band_6h[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals