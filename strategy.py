#!/usr/bin/env python3
name = "6h_Retracement_Liquidity_Trap"
timeframe = "6h"
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
    
    # 1d data for swing detection (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    prev_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    
    # Daily range for liquidity trap detection
    daily_range = prev_high_1d - prev_low_1d
    range_mid = prev_low_1d + (daily_range * 0.5)
    
    # 12h data for trend filter (EMA20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter (3-period average)
    vol_ma = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    # Align daily-based levels to 6h
    range_mid_aligned = align_htf_to_ltf(prices, df_1d, range_mid)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(2, 20, 3)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(range_mid_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price approaches daily midpoint from below with volume, above 12h EMA20
            if (close[i] > range_mid_aligned[i] * 0.995 and  # Near midpoint from below
                close[i-1] <= range_mid_aligned[i-1] * 0.995 and  # Was below
                volume[i] > vol_ma[i] * 1.5 and  # Volume spike
                close[i] > ema_20_12h_aligned[i]):  # Above trend
                signals[i] = 0.25
                position = 1
            # Short: Price approaches daily midpoint from above with volume, below 12h EMA20
            elif (close[i] < range_mid_aligned[i] * 1.005 and  # Near midpoint from above
                  close[i-1] >= range_mid_aligned[i-1] * 1.005 and  # Was above
                  volume[i] > vol_ma[i] * 1.5 and  # Volume spike
                  close[i] < ema_20_12h_aligned[i]):  # Below trend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price moves above midpoint or below 12h EMA20
            if close[i] > range_mid_aligned[i] * 1.01 or close[i] < ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price moves below midpoint or above 12h EMA20
            if close[i] < range_mid_aligned[i] * 0.99 or close[i] > ema_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals