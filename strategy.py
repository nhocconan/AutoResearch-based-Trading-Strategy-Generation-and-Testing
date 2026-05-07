#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses 12h price channels (Donchian) for breakouts, filtered by 1d EMA trend direction
# and confirmed by volume spike. Designed to work in both bull and bear markets by
# only taking long positions in uptrends and short positions in downtrends.
# Target: 12-30 trades/year per symbol to avoid fee drag on 12h timeframe.
name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average for spike detection
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 12h Donchian channels (20-period high/low)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume spike: current volume > 1.5x 1d EMA volume average
        vol_spike = volume[i] > (vol_avg_1d_aligned[i] * 1.5)
        
        if position == 0:
            # Long breakout: price > 12h Donchian high with volume spike in uptrend
            long_condition = (high[i] > highest_high[i-1]) and vol_spike and uptrend
            # Short breakdown: price < 12h Donchian low with volume spike in downtrend
            short_condition = (low[i] < lowest_low[i-1]) and vol_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below Donchian high or trend turns down
            if (close[i] < highest_high[i-1]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above Donchian low or trend turns up
            if (close[i] > lowest_low[i-1]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals