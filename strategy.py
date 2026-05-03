#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50 for long, < for short) and volume confirmation.
# Weekly EMA50 provides robust trend filter that adapts to both bull and bear markets.
# Donchian breakouts capture momentum; volume spike confirms validity.
# Target: 75-150 total trades over 4 years (19-37/year) to stay within fee limits.
# Uses discrete sizing 0.25 to minimize churn. Works on BTC/ETH/SOL.

name = "6h_Donchian20_WeeklyEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # weekly uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # weekly downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Donchian high AND weekly uptrend AND volume spike
            if close_val > highest_20[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly downtrend AND volume spike
            elif close_val < lowest_20[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR weekly trend turns down
            if close_val < lowest_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR weekly trend turns up
            if close_val > highest_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals