#!/usr/bin/env python3
name = "6h_RVOL_Breakout_MultiTF_Trend"
timeframe = "6h"
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
    
    # === 12h RVOL (relative volume) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    rvol_12h = vol_12h / vol_avg_12h
    rvol_12h_aligned = align_htf_to_ltf(prices, df_12h, rvol_12h)
    
    # === 1d Close trend filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h Donchian(20) breakout levels ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rvol_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(donch_high_12h_aligned[i]) or
            np.isnan(donch_low_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above 12h Donchian high + RVOL > 1.5 + close above 1d EMA34
            if (close[i] > donch_high_12h_aligned[i] and
                rvol_12h_aligned[i] > 1.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below 12h Donchian low + RVOL > 1.5 + close below 1d EMA34
            elif (close[i] < donch_low_12h_aligned[i] and
                  rvol_12h_aligned[i] > 1.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below 12h Donchian low or below 1d EMA34
            if close[i] < donch_low_12h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above 12h Donchian high or above 1d EMA34
            if close[i] > donch_high_12h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals