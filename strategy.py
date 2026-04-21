#!/usr/bin/env python3
"""
1d_1w_Donchian20_VolumeBreakout_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high with weekly close > weekly EMA34 and volume spike.
Short when price breaks below 20-day low with weekly close < weekly EMA34 and volume spike.
Exit via ATR-based trailing stop (3*ATR) or opposite breakout.
Works in bull via breakouts, in bear via short breakdowns. Volume filter reduces false signals.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Donchian and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian(20)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align to 1d timeframe (prices is already 1d)
    highest_20_aligned = highest_20  # no alignment needed for same timeframe
    lowest_20_aligned = lowest_20
    atr_aligned = atr
    
    # Load weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above 20-day high with weekly uptrend and volume
            if (price > highest_20_aligned[i] and 
                close_1d[i] > ema_34_1w_aligned[i] and  # weekly close above EMA34
                volume_ok):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short conditions: break below 20-day low with weekly downtrend and volume
            elif (price < lowest_20_aligned[i] and 
                  close_1d[i] < ema_34_1w_aligned[i] and  # weekly close below EMA34
                  volume_ok):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 3*ATR from highest
            if price <= highest_since_entry - 3.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Opposite breakout exit: break below 20-day low
            elif price < lowest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 3*ATR from lowest
            if price >= lowest_since_entry + 3.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Opposite breakout exit: break above 20-day high
            elif price > highest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian20_VolumeBreakout_ATRStop_v1"
timeframe = "1d"
leverage = 1.0