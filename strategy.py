#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRFilter_V2
Hypothesis: Donchian(20) breakout on 4h with volume confirmation and ATR-based stoploss.
Works in bull/bear: Breakouts capture strong moves in both directions. Volume filter ensures participation.
ATR stoploss limits downside. Target: 20-40 trades/year per symbol (80-160 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (HTF = 12h as per experiment)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 4h (primary timeframe)
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Calculate ATR(14) for stoploss and volume filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend (12h EMA50 rising) + volume
            if (price > upper[i] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend (12h EMA50 falling) + volume
            elif (price < lower[i] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12h EMA50 or ATR-based stop
            if (price < ema_50_12h_aligned[i] or 
                price < prices['close'].iloc[i-1] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h EMA50 or ATR-based stop
            if (price > ema_50_12h_aligned[i] or 
                price > prices['close'].iloc[i-1] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter_V2"
timeframe = "4h"
leverage = 1.0