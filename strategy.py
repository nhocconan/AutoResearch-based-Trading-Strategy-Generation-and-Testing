#!/usr/bin/env python3
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
    
    # === 1d data (HTF for bias and volatility) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d EMA 34 for trend bias ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d ATR for volatility regime ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6h Donchian Channel (16) ===
    highest_16 = pd.Series(high).rolling(window=16, min_periods=16).max().values
    lowest_16 = pd.Series(low).rolling(window=16, min_periods=16).min().values
    
    # === 6h Volume spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    warmup = 100
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(highest_16[i]) or np.isnan(lowest_16[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        price = close[i]
        bias = ema_34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit logic
        if position == 1:
            if price < lowest_16[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if price > highest_16[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic
        if position == 0 and in_session:
            # Long: price above EMA34 (bullish bias), break above Donchian, volume spike
            if (price > bias) and (price > highest_16[i]) and (vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA34 (bearish bias), break below Donchian, volume spike
            elif (price < bias) and (price < lowest_16[i]) and (vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA34_Donchian_Breakout_Volume_Session"
timeframe = "6h"
leverage = 1.0