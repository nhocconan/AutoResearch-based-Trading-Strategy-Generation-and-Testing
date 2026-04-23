#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike
- Uses Donchian channel breakout for trend entry, confirmed by 1d ATR volatility regime and volume spike
- Designed for 4h timeframe to capture medium-term moves with controlled trade frequency
- ATR filter ensures we only trade when volatility is elevated (avoids choppy low-vol periods)
- Volume confirmation adds conviction to breakouts
- Target: 20-50 trades/year per symbol to stay within fee drag limits
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
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian(20) channels on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period MA (elevated vol)
        atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        vol_filter = atr_14_1d_aligned[i] > atr_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band + volume spike + vol filter
            if (close[i] > highest_high[i] and 
                volume[i] > 1.3 * vol_ma[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band + volume spike + vol filter
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.3 * vol_ma[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel OR volatility drops
            middle = (highest_high[i] + lowest_low[i]) / 2
            exit_signal = False
            
            if position == 1:
                # Exit long when price falls below middle OR volatility drops
                if close[i] < middle or not vol_filter:
                    exit_signal = True
            elif position == -1:
                # Exit short when price rises above middle OR volatility drops
                if close[i] > middle or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Volume_Breakout"
timeframe = "4h"
leverage = 1.0