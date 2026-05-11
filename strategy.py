#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
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
    
    # 1d data for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_hl = high_1d - low_1d
    # R1 = Pivot + (Range * 1.1 / 12)
    R1 = pivot + (range_hl * 1.1 / 12)
    # S1 = Pivot - (Range * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # Align R1 and S1 to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 34-period EMA for trend filter (1d EMA34)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detector: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1, uptrend (price > EMA34), volume spike
            if (close[i] > R1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, downtrend (price < EMA34), volume spike
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S1 (opposite level) or trend breaks
            if (close[i] < S1_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R1 (opposite level) or trend breaks
            if (close[i] > R1_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals