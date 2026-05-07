#!/usr/bin/env python3
name = "1d_Donchian_20_Weekly_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) breakout
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above Donchian high in weekly uptrend with volume
            if close[i] > donchian_high[i] and close[i] > ema_20_1d[i] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low in weekly downtrend with volume
            elif close[i] < donchian_low[i] and close[i] < ema_20_1d[i] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or trend reverses
            if close[i] < donchian_low[i] or close[i] < ema_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns to Donchian high or trend reverses
            if close[i] > donchian_high[i] or close[i] > ema_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals