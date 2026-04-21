#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_v2
Hypothesis: Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
Long when price breaks above upper Donchian with 1d EMA50 rising and volume spike.
Short when price breaks below lower Donchian with 1d EMA50 falling and volume spike.
Exit on opposite Donchian break or ATR-based stoploss.
Works in both bull/bear by following 1d trend and using volatility-based breakouts.
Target: 20-35 trades/year per symbol.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 4h prices
    high_roll = prices['high'].rolling(window=20, min_periods=20).max().values
    low_roll = prices['low'].rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr_multiplier = 2.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Calculate ATR(14) for stoploss
        if i >= 14:
            tr1 = prices['high'].iloc[i-14:i] - prices['low'].iloc[i-14:i]
            tr2 = abs(prices['high'].iloc[i-14:i] - prices['close'].iloc[i-15:i-1])
            tr3 = abs(prices['low'].iloc[i-14:i] - prices['close'].iloc[i-15:i-1])
            tr = np.maximum.reduce([tr1, tr2, tr3])
            atr = tr.mean()
        else:
            atr = 0.0
        
        if position == 0:
            # Long conditions: break above upper Donchian with bullish trend and volume
            if (price > high_roll[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # EMA50 rising
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower Donchian with bearish trend and volume
            elif (price < low_roll[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # EMA50 falling
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below lower Donchian or ATR stop
            if price < low_roll[i] or (i >= 1 and prices['close'].iloc[i-1] > 0 and 
                                       price < prices['close'].iloc[i-1] - atr_multiplier * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above upper Donchian or ATR stop
            if price > high_roll[i] or (i >= 1 and prices['close'].iloc[i-1] > 0 and 
                                        price > prices['close'].iloc[i-1] + atr_multiplier * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_v2"
timeframe = "4h"
leverage = 1.0