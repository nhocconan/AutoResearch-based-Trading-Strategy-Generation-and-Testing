#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>1.6x 20-bar avg volume).
# Uses Donchian channel for structure, 1w EMA50 for strong trend alignment (avoids counter-trend trades),
# and volume spike to confirm institutional interest. Designed for low trade frequency (target: 50-150 total over 4 years)
# to minimize fee drag while capturing sustained moves in both bull and bear markets. Exit on reverse Donchian touch.

name = "12h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period) for entry/exit
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 20), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 1d Donchian high, close > 1w EMA50, volume spike (>1.6x avg)
            if (high[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.6 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 1d Donchian low, close < 1w EMA50, volume spike (>1.6x avg)
            elif (low[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.6 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below 1d Donchian low
            if low[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above 1d Donchian high
            if high[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals