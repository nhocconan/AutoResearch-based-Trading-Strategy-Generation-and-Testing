#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian(20) breakouts capture strong momentum moves in both bull and bear markets
# 1w EMA50 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation (1.5x 20-period average) filters out low-conviction breakouts
# Exit on Donchian(10) opposite breakout or close below/above 1w EMA50
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20 for entry, 10 for exit)
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_roll_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_roll_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)  # warmup for EMA50, Donchian20, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_roll_20[i]) or 
            np.isnan(low_roll_20[i]) or np.isnan(high_roll_10[i]) or 
            np.isnan(low_roll_10[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_donchian_high_20 = high_roll_20[i]
        curr_donchian_low_20 = low_roll_20[i]
        curr_donchian_high_10 = high_roll_10[i]
        curr_donchian_low_10 = low_roll_10[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above Donchian(20) high AND above 1w EMA50 (uptrend)
                if curr_close > curr_donchian_high_20 and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below Donchian(20) low AND below 1w EMA50 (downtrend)
                elif curr_close < curr_donchian_low_20 and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: 
            # 1. Close below Donchian(10) low (breakout fails)
            # 2. Close below 1w EMA50 (trend change)
            if curr_close < curr_donchian_low_10 or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Close above Donchian(10) high (breakdown fails)
            # 2. Close above 1w EMA50 (trend change)
            if curr_close > curr_donchian_high_10 or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals