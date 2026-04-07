#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v3
Hypothesis: Focus on high-probability breakouts at S4/R4 with strong volume and trend confirmation.
Reduce overtrading by requiring both volume spike (>2x average) and strong trend (price > EMA50 for longs, < EMA50 for shorts).
Only trade in direction of daily trend to avoid counter-trend whipsaws. Target 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align daily levels to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume (strict)
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S4 OR EMA turns down
            if close[i] < s4_4h[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R4 OR EMA turns up
            if close[i] > r4_4h[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout longs at R4 in uptrend (price > EMA)
            if (close[i] >= r4_4h[i] and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Breakout shorts at S4 in downtrend (price < EMA)
            elif (close[i] <= s4_4h[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals