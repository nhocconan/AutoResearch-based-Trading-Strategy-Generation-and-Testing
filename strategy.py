#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 Trend Filter and Volume Spike
- Uses Camarilla pivot levels (R3/S3) from daily timeframe for structure-based breakouts
- 12h EMA50 defines higher timeframe trend filter: only trade breakouts in direction of 12h trend
- Volume confirmation (> 2.0x 20-period average) filters weak signals
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading structure breaks with trend alignment
"""

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
    
    # Calculate daily Camarilla pivot levels (R3/S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    camarilla_r3 = close_1d + camarilla_range
    camarilla_s3 = close_1d - camarilla_range
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND above 12h EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND below 12h EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Camarilla pivot center (close) OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price closes below daily close (pivot point) OR price closes below 12h EMA50
                if (close[i] < close_1d[-1] if i == len(prices)-1 else close[i] < np.interp(i, 
                           np.arange(len(close_1d))*len(prices)/len(close_1d), close_1d) or 
                    close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price closes above daily close (pivot point) OR price closes above 12h EMA50
                if (close[i] > close_1d[-1] if i == len(prices)-1 else close[i] > np.interp(i, 
                           np.arange(len(close_1d))*len(prices)/len(close_1d), close_1d) or 
                    close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0