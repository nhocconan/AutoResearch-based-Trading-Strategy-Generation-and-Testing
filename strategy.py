#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with weekly pivot direction filter and volume confirmation.
# Uses Donchian breakouts (20-period) for trend continuation, filtered by weekly pivot levels to ensure
# alignment with higher timeframe structure, and volume spikes to confirm breakout strength.
# Designed to work in both bull and bear markets by following weekly pivot trend direction.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to avoid excessive fee drag.
name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot direction filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Weekly pivot (using prior week) - acts as trend filter
    high_prev_w = df_w['high'].shift(1).values
    low_prev_w = df_w['low'].shift(1).values
    close_prev_w = df_w['close'].shift(1).values
    pivot_w = (high_prev_w + low_prev_w + close_prev_w) / 3
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    
    # Donchian channel (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection (20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to weekly pivot
        above_pivot = close[i] > pivot_w_aligned[i]
        below_pivot = close[i] < pivot_w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume spike and above weekly pivot
            long_condition = (high[i] > highest_high[i-1]) and vol_spike[i] and above_pivot
            # Short breakdown: price breaks below Donchian low with volume spike and below weekly pivot
            short_condition = (low[i] < lowest_low[i-1]) and vol_spike[i] and below_pivot
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below Donchian high or breaks below weekly pivot
            if (close[i] < highest_high[i-1]) or (not above_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above Donchian low or breaks above weekly pivot
            if (close[i] > lowest_low[i-1]) or (not below_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals