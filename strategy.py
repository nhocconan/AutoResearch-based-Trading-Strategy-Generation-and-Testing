#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate weekly Donchian channels (20-period)
    # For weekly data, 20 periods = 20 weeks
    donchian_high_20w = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low_20w = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_20w_6h = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_20w_6h = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    
    # Get weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike: current volume > 2.0 * 8-period average (48h lookback)
    vol_ma_8 = np.full(n, np.nan)
    for i in range(8, n):
        vol_ma_8[i] = np.mean(volume[i-8:i])
    volume_spike = volume > (2.0 * vol_ma_8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(20, 8) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_20w_6h[i]) or np.isnan(donchian_low_20w_6h[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_8[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + weekly uptrend + volume spike
            if (close[i] > donchian_high_20w_6h[i] and close[i] > ema20_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low + weekly downtrend + volume spike
            elif (close[i] < donchian_low_20w_6h[i] and close[i] < ema20_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low (reversal) or trend changes
            if (close[i] < donchian_low_20w_6h[i] or close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high (reversal) or trend changes
            if (close[i] > donchian_high_20w_6h[i] or close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20w_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0