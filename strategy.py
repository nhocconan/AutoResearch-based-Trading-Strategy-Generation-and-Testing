#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeS
# Hypothesis: 4h strategy using Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with volume > 1.5x average and price > 1d EMA34.
# Short when price breaks below Camarilla S3 with volume > 1.5x average and price < 1d EMA34.
# Exit on opposite Camarilla level (S3 for long, R3 for short) or trend reversal.
# Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing institutional levels.

timeframe = "4h"
name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeS"
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
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    # Camarilla: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*0.5/2
    #          S1 = C - (H-L)*0.5/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # Where C = (H+L+C)/3 of previous day
    # We need previous day's OHLC, so shift by 1
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].values[:-1]  # previous day's high
    prev_low = df_1d['low'].values[:-1]    # previous day's low
    prev_close = df_1d['close'].values[:-1] # previous day's close
    
    # Calculate pivot and ranges
    prev_pivot = (prev_high + prev_low + prev_close) / 3.0
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_pivot + (prev_range * 1.25 / 2)
    S3 = prev_pivot - (prev_range * 1.25 / 2)
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike detection: 1.5x average volume (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume and above daily EMA34
            if (high[i] > R3_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and below daily EMA34
            elif (low[i] < S3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 (failed breakout) or trend reversal
            if low[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 (failed breakdown) or trend reversal
            if high[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals