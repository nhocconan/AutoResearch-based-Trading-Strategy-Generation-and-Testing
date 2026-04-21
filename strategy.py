#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_Trend_ATRStop_V1
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
Works in bull/bear by using 1d EMA50 for trend direction and Donchian channels for structure.
Volume confirmation added to reduce false breakouts. Target: 12-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data once for EMA trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for stoploss and position sizing
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Load 12h data for Donchian channels (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian(20) channels
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    
    # 12h volume confirmation (20-period average)
    volume_12h = df_12h['volume'].values
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(volume_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_ok = volume > 1.5 * volume_ma_20_aligned[i]
        
        if position == 0:
            # Long conditions: break above Donchian upper with 1d uptrend and volume
            if (price > highest_20_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # 1d EMA rising
                volume_ok):
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_14_1d_aligned[i]
            # Short conditions: break below Donchian lower with 1d downtrend and volume
            elif (price < lowest_20_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # 1d EMA falling
                  volume_ok):
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_14_1d_aligned[i]
        
        elif position == 1:
            # Long exit: stoploss hit or Donchian lower break
            if price <= atr_stop or price < lowest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stoploss up: move stop to breakeven after 1.5*ATR profit
                if price >= entry_price + 1.5 * atr_14_1d_aligned[i]:
                    atr_stop = max(atr_stop, entry_price)
        
        elif position == -1:
            # Short exit: stoploss hit or Donchian upper break
            if price >= atr_stop or price > highest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stoploss down: move stop to breakeven after 1.5*ATR profit
                if price <= entry_price - 1.5 * atr_14_1d_aligned[i]:
                    atr_stop = min(atr_stop, entry_price)
    
    return signals

name = "12h_1d_Donchian20_Breakout_Trend_ATRStop_V1"
timeframe = "12h"
leverage = 1.0