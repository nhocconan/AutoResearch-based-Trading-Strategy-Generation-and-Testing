#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 (using previous day's range)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike (24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1, above 12h EMA50, volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1, below 12h EMA50, volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S1 or below 12h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R1 or above 12h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals