#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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
    
    # Daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla pivot levels from previous day
    close_prev = close_1d[-1] if len(close_1d) > 0 else 0
    high_prev = high_1d[-1] if len(high_1d) > 0 else 0
    low_prev = low_1d[-1] if len(low_1d) > 0 else 0
    range_prev = high_prev - low_prev
    
    # Calculate Camarilla levels for current day based on previous day
    R4 = close_prev + range_prev * 1.500
    R3 = close_prev + range_prev * 1.250
    R2 = close_prev + range_prev * 1.166
    R1 = close_prev + range_prev * 1.083
    S1 = close_prev - range_prev * 1.083
    S2 = close_prev - range_prev * 1.166
    S3 = close_prev - range_prev * 1.250
    S4 = close_prev - range_prev * 1.500
    
    # Arrays for full length (same value for all intraday bars of the day)
    R1_arr = np.full_like(close_1d, R1)
    S1_arr = np.full_like(close_1d, S1)
    
    # EMA(34) on daily for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1_arr)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1_arr)
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike on 4h: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema34_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + price > EMA34 (uptrend) + volume spike
            if (close[i] > R1_4h[i] and 
                close[i] > ema34_4h[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + price < EMA34 (downtrend) + volume spike
            elif (close[i] < S1_4h[i] and 
                  close[i] < ema34_4h[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 or price < EMA34
            if close[i] < S1_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 or price > EMA34
            if close[i] > R1_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals