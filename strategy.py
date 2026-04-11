#!/usr/bin/env python3
# 1h_4h_donchian_volume_v1
# Strategy: 1h Donchian breakout with volume confirmation and 4h EMA trend filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: 1h Donchian(20) breakouts with volume confirmation and 4h EMA(20) trend filter work in both bull and bear markets by capturing strong directional moves while avoiding whipsaws. Uses 4h for direction, 1h for timing. Target: 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_donchian_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(high_20.iloc[i]) or 
            np.isnan(low_20.iloc[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        if vol_confirmed and close[i] > high_20.iloc[i-1] and close[i] > ema_20_4h_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.20
        elif vol_confirmed and close[i] < low_20.iloc[i-1] and close[i] < ema_20_4h_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: trend reversal or opposite breakout
        elif position == 1 and (close[i] < ema_20_4h_aligned[i] or close[i] < low_20.iloc[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_20_4h_aligned[i] or close[i] > high_20.iloc[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals