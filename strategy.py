#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v6"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla levels
    prev_close = close_1d[-1] if len(close_1d) > 0 else 0
    prev_high = high_1d[-1] if len(high_1d) > 0 else 0
    prev_low = low_1d[-1] if len(low_1d) > 0 else 0
    range_val = prev_high - prev_low
    
    # Camarilla levels (using previous day's data)
    R4 = prev_close + range_val * 1.1 / 2
    R3 = prev_close + range_val * 1.1 / 4
    R2 = prev_close + range_val * 1.1 / 6
    R1 = prev_close + range_val * 1.1 / 12
    S1 = prev_close - range_val * 1.1 / 12
    S2 = prev_close - range_val * 1.1 / 6
    S3 = prev_close - range_val * 1.1 / 4
    S4 = prev_close - range_val * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA34 to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, R1))
    S1_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, S1))
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h EMA13 for entry timing
    ema13_4h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 35  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13_4h[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and uptrend (price > EMA34)
            if (close[i] > R1_aligned[i] and 
                vol_ratio[i] > 1.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and downtrend (price < EMA34)
            elif (close[i] < S1_aligned[i] and 
                  vol_ratio[i] > 1.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 or trend reverses
            if (close[i] < S1_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 or trend reverses
            if (close[i] > R1_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals