#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_20_Trend_Filter"
timeframe = "1d"
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
    
    # === Weekly Donchian (20) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper/lower bands (20-period)
    donch_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower)
    
    # === Weekly EMA20 trend filter ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Daily volume spike filter ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above weekly Donchian upper + above weekly EMA20 + volume spike
            if (close[i] > donch_upper_aligned[i] and
                close[i] > ema20_1w_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly Donchian lower + below weekly EMA20 + volume spike
            elif (close[i] < donch_lower_aligned[i] and
                  close[i] < ema20_1w_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly Donchian lower or below weekly EMA20
            if close[i] < donch_lower_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly Donchian upper or above weekly EMA20
            if close[i] > donch_upper_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals