#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend_ATRFilter_v1
Hypothesis: Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation on 4h timeframe.
Works in bull/bear: Long when price breaks above 20-period high AND 1d EMA50 rising AND volume spike.
Short when price breaks below 20-period low AND 1d EMA50 falling AND volume spike.
Uses ATR-based stoploss via signal=0 when price moves against position by 2*ATR.
Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss and volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: price > Donchian high AND 1d EMA50 rising AND volume spike
            if (price > highest_high[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short conditions: price < Donchian low AND 1d EMA50 falling AND volume spike
            elif (price < lowest_low[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stoploss or trend reversal
            if price < entry_price - 2.0 * atr[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stoploss or trend reversal
            if price > entry_price + 2.0 * atr[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0