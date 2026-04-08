#!/usr/bin/env python3
"""
12h_bb_reversion_v1
Hypothesis: Mean reversion at Bollinger Bands with 1d trend filter on 12h timeframe.
- Uses Bollinger Bands (20,2) on 12h for overbought/oversold signals
- 1d EMA50 filter ensures trades align with higher timeframe trend
- Volume confirmation (1.5x average) filters false signals
- Targets 15-25 trades/year to stay within fee limits
- Works in both bull and bear markets by trading reversions within trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_bb_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 12h (20,2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Volume average (20-period)
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=20, min_periods=20).mean()
    
    # Daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price touches middle band or trend turns bearish
            if close[i] >= basis[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price touches middle band or trend turns bullish
            if close[i] <= basis[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: oversold + volume spike + uptrend
            if (close[i] <= lower[i] and 
                volume[i] > 1.5 * vol_ma[i] and
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: overbought + volume spike + downtrend
            elif (close[i] >= upper[i] and 
                  volume[i] > 1.5 * vol_ma[i] and
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals