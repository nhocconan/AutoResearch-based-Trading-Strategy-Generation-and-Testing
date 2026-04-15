#!/usr/bin/env python3
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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    highest_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian high (breakout)
        # 2. Price above weekly EMA50 (bullish bias)
        if (close[i] > highest_20_aligned[i] and
            close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian low (breakdown)
        # 2. Price below weekly EMA50 (bearish bias)
        elif (close[i] < lowest_20_aligned[i] and
              close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WeeklyDonchian20_EMA50_Breakout_v1"
timeframe = "12h"
leverage = 1.0