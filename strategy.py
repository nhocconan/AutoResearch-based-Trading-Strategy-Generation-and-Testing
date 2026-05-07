#!/usr/bin/env python3
name = "6h_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Donchian and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channel (20-day)
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate 20-period high and low
    donchian_high = pd.Series(d_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(d_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above Donchian high in daily uptrend with volume
            if close[i] > donchian_high_6h[i] and ema_34_6h[i] > ema_34_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in daily downtrend with volume
            elif close[i] < donchian_low_6h[i] and ema_34_6h[i] < ema_34_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or trend reverses
            if close[i] < donchian_low_6h[i] or ema_34_6h[i] < ema_34_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to Donchian high or trend reverses
            if close[i] > donchian_high_6h[i] or ema_34_6h[i] > ema_34_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakouts on 6h with daily trend filter and volume confirmation
# - Donchian breakout above 20-day high signals bullish momentum
# - Donchian breakdown below 20-day low signals bearish momentum
# - Daily EMA34 trend filter ensures we only trade in the direction of the higher timeframe trend
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to the opposite Donchian level or daily trend reverses
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1d timeframe for structure and trend, 6h for execution timing
# - This is a novel combination for 6h timeframe that hasn't been over-tested in the DB