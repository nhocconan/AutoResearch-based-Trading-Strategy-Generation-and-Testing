#!/usr/bin/env python3
"""
4h_LinearRegressionTrend_1dATR_Stop
Hypothesis: Linear regression slope on 1-day closes indicates long-term trend direction.
Price crossing above/below 1-day linear regression channel (mean ± 1*ATR) with volume confirmation
captures trend continuation moves. ATR-based stop loss manages risk. Designed for 4h to achieve
20-35 trades/year with strong trend capture in both bull and bear markets by following 1d trend.
"""
name = "4h_LinearRegressionTrend_1dATR_Stop"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day linear regression slope and intercept (20-period)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Linear regression: y = mx + b over last 20 days
    def linreg_slope(arr, lookback=20):
        if len(arr) < lookback:
            return np.nan
        y = arr[-lookback:]
        x = np.arange(lookback)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x * x).sum()
        slope = (lookback * sum_xy - sum_x * sum_y) / (lookback * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / lookback
        return slope, intercept
    
    # Calculate ATR for volatility bands
    def calculate_atr(high_arr, low_arr, close_arr, lookback=14):
        if len(high_arr) < lookback + 1:
            return np.full_like(high_arr, np.nan)
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=lookback, adjust=False).mean().values
        return atr
    
    # Pre-calculate linear regression channels
    lr_slope = np.full(len(close_1d), np.nan)
    lr_intercept = np.full(len(close_1d), np.nan)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    start_idx = 20  # Need 20 days for regression
    for i in range(start_idx, len(close_1d)):
        slope, intercept = linreg_slope(close_1d[:i+1], 20)
        lr_slope[i] = slope
        lr_intercept[i] = intercept
    
    # Calculate upper and lower bands: mean ± 1*ATR
    lr_mean = lr_slope * np.arange(len(close_1d)) + lr_intercept
    upper_band = lr_mean + atr_1d
    lower_band = lr_mean - atr_1d
    
    # Align to 4h timeframe
    lr_slope_aligned = align_htf_to_ltf(prices, df_1d, lr_slope)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(lr_slope_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper band + positive slope + volume
            if (close[i] > upper_band_aligned[i] and 
                lr_slope_aligned[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lower band + negative slope + volume
            elif (close[i] < lower_band_aligned[i] and 
                  lr_slope_aligned[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite band or slope changes sign
            if position == 1:
                if close[i] < lower_band_aligned[i] or lr_slope_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_band_aligned[i] or lr_slope_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals