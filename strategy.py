#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) volatility filter.
Long when price breaks above upper Donchian channel AND close > 1d EMA50 AND ATR ratio > 0.8.
Short when price breaks below lower Donchian channel AND close < 1d EMA50 AND ATR ratio > 0.8.
Exit when price reverts to middle Donchian channel or ATR ratio drops below 0.5 (low volatility).
Uses 4h timeframe targeting 75-200 total trades over 4 years. Donchian provides clear structure,
1d EMA50 filters for higher timeframe trend, ATR ratio ensures trades occur during sufficient volatility.
Works in both bull and bear markets by aligning with higher timeframe direction and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_dc = (highest_high + lowest_low) / 2.0
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR average) for volatility regime filter
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 0.0)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_dc = highest_high[i]
        lower_dc = lowest_low[i]
        middle = middle_dc[i]
        ema50_val = ema50_1d_aligned[i]
        volatility = atr_ratio[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1d EMA50 AND sufficient volatility
            if (price > upper_dc and price > ema50_val and volatility > 0.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 1d EMA50 AND sufficient volatility
            elif (price < lower_dc and price < ema50_val and volatility > 0.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle Donchian OR volatility drops too low
                if price < middle or volatility < 0.5:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle Donchian OR volatility drops too low
                if price > middle or volatility < 0.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_ATR_Volatility_Filter"
timeframe = "4h"
leverage = 1.0