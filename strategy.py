#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_1wFilter"
timeframe = "12h"
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
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1w filter: EMA150 (long-term trend)
    df_1w = get_htf_data(prices, '1w')
    ema150_1w = pd.Series(df_1w['close'].values).ewm(span=150, adjust=False, min_periods=150).mean().values
    ema150_1w_aligned = align_htf_to_ltf(prices, df_1w, ema150_1w)
    
    # 12h Donchian channels (20-period)
    lookback = 20
    high_roll = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_roll = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # wait for enough data
    
    for i in range(start_idx, n):
        # Skip if 1d or 1w trend data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema150_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + 1d uptrend + 1w uptrend
            if (close[i] > high_roll[i] and 
                close[i] > ema50_1d_aligned[i] and 
                close[i] > ema150_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + 1d downtrend + 1w downtrend
            elif (close[i] < low_roll[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  close[i] < ema150_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price closes below Donchian low or 1d trend turns down
            if (close[i] < low_roll[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price closes above Donchian high or 1d trend turns up
            if (close[i] > high_roll[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals