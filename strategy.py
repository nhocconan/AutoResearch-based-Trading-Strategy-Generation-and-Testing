#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with weekly pivot direction filter and volume confirmation.
# Uses weekly Donchian(20) breakout confirmed by weekly pivot point trend (above/below weekly pivot) and 6h volume spikes.
# Weekly pivot provides long-term trend filter that adapts to bull/bear markets, reducing false breakouts.
# Target: 15-35 trades/year per symbol to minimize fee drag while capturing significant moves.
name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian and pivot
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_max = df_w['high'].rolling(window=20, min_periods=20).max().values
    low_min = df_w['low'].rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point from previous week
    high_prev_w = df_w['high'].shift(1).values
    low_prev_w = df_w['low'].shift(1).values
    close_prev_w = df_w['close'].shift(1).values
    weekly_pivot = (high_prev_w + low_prev_w + close_prev_w) / 3
    
    # Align weekly indicators to 6s timeframe
    donchian_high = align_htf_to_ltf(prices, df_w, high_max)
    donchian_low = align_htf_to_ltf(prices, df_w, low_min)
    pivot_w = align_htf_to_ltf(prices, df_w, weekly_pivot)
    
    # 6h volume average for spike detection (20-period EMA)
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_w[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly pivot
        above_pivot = close[i] > pivot_w[i]
        below_pivot = close[i] < pivot_w[i]
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume spike and above weekly pivot
            long_breakout = close[i] > donchian_high[i]
            long_condition = long_breakout and vol_spike[i] and above_pivot
            
            # Short breakdown: price breaks below weekly Donchian low with volume spike and below weekly pivot
            short_breakout = close[i] < donchian_low[i]
            short_condition = short_breakout and vol_spike[i] and below_pivot
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below weekly Donchian high or falls below weekly pivot
            if close[i] < donchian_high[i] or close[i] < pivot_w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above weekly Donchian low or rises above weekly pivot
            if close[i] > donchian_low[i] or close[i] > pivot_w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals