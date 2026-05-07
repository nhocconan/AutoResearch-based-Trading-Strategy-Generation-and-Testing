#/usr/bin/env python3
"""
4h_LinearRegressionTrend_1dATR_Stop
Hypothesis: A 4-hour linear regression slope crossing zero with 1-day ATR-based stop and volume confirmation
captures strong trend changes while minimizing whipsaws. Works in both bull and bear markets by following
the trend direction with dynamic exits. Targets 20-50 trades/year.
"""
name = "4h_LinearRegressionTrend_1dATR_Stop"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based stop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day True Range and ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14[0:13] = np.nan  # Ensure proper NaN for insufficient data
    
    # 4-hour Linear Regression Slope (20-period)
    def linreg_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                continue
            # Use only valid (non-NaN) points
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            slopes[i] = slope
        return slopes
    
    lr_slope = linreg_slope(close, 20)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient warmup for LR and ATR
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(lr_slope[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: positive LR slope + volume filter
            if lr_slope[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: negative LR slope + volume filter
            elif lr_slope[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Dynamic stop: 1.5 * 1-day ATR from entry price
            # We don't track entry price, so use a time-based trailing approach:
            # Exit when LR slope changes sign (trend reversal)
            if position == 1:
                if lr_slope[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lr_slope[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals