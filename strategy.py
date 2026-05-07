#!/usr/bin/env python3
"""
4H_Camarilla_R3S3_Breakout_1dEMA34_Volume
Hypothesis: On 4h timeframe, use Camarilla R3/S3 levels from prior 1d as breakout levels with 1d EMA34 trend filter and volume confirmation. This uses price structure (Camarilla pivots) for institutional-level entries, EMA34 for trend alignment, and volume to filter false breakouts. Targets 20-35 trades/year.
"""
name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC (use previous day's data to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)  # Previous day's close
    high_1d_shifted = np.roll(high_1d, 1)   # Previous day's high
    low_1d_shifted = np.roll(low_1d, 1)     # Previous day's low
    # Set first element to NaN since no prior day exists
    close_1d_shifted[0] = np.nan
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    
    # Camarilla R3 and S3 levels from prior day
    # R3 = Close + 1.1*(High - Low)/2
    # S3 = Close - 1.1*(High - Low)/2
    camarilla_r3 = close_1d_shifted + 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    camarilla_s3 = close_1d_shifted - 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current 4h volume > 1.3 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above Camarilla R3, above 1d EMA34, and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below Camarilla S3, below 1d EMA34, and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla S3 (reversion to mean)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla R3 (reversion to mean)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals