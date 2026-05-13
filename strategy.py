#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Camarilla levels, trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels
    def calculate_camarilla(high, low, close):
        # Previous day's values
        ph = np.concatenate([[np.nan], high[:-1]])
        pl = np.concatenate([[np.nan], low[:-1]])
        pc = np.concatenate([[np.nan], close[:-1]])
        
        # Camarilla levels
        R1 = pc + 1.1 * (ph - pl) / 12
        S1 = pc - 1.1 * (ph - pl) / 12
        R2 = pc + 1.1 * (ph - pl) / 6
        S2 = pc - 1.1 * (ph - pl) / 6
        R3 = pc + 1.1 * (ph - pl) / 4
        S3 = pc - 1.1 * (ph - pl) / 4
        R4 = pc + 1.1 * (ph - pl) / 2
        S4 = pc - 1.1 * (ph - pl) / 2
        
        return R1, S1, R2, S2, R3, S3, R4, S4
    
    R1, S1, R2, S2, R3, S3, R4, S4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1D volume average for volume spike filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1D indicators to 4H timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * vol_avg_1d_aligned[i]
        
        if position == 0:
            # LONG: price breaks above R1 + uptrend + volume spike
            if close[i] > R1_aligned[i] and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + downtrend + volume spike
            elif close[i] < S1_aligned[i] and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend changes
            if close[i] < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend changes
            if close[i] > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals