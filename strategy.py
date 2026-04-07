#!/usr/bin/env python3
"""
12h Supertrend with Daily Trend Filter and Volume Confirmation
Long when Supertrend flips below price AND daily close > daily EMA50 with volume > 1.5x average
Short when Supertrend flips above price AND daily close < daily EMA50 with volume > 1.5x average
Exit when Supertrend flips opposite direction
Designed to work in both bull and bear markets by following the higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_supertrend_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Supertrend (12h) ===
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    uptrend = True  # True for uptrend, False for downtrend
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1] if i > 0 else hl2[i]
            continue
            
        if close[i] > upper[i-1]:
            uptrend = True
        elif close[i] < lower[i-1]:
            uptrend = False
        else:
            uptrend = uptrend
            
        if uptrend:
            supertrend[i] = max(lower[i], supertrend[i-1] if i > 0 else lower[i])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1] if i > 0 else upper[i])
    
    # === Daily Trend Filter (1d) ===
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate daily EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    # Align to 12h timeframe
    daily_trend = align_htf_to_ltf(prices, df_1d, ema_50)
    daily_close = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # === Volume confirmation (12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(supertrend[i]) or np.isnan(daily_trend[i]) or np.isnan(daily_close[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Supertrend flips above price (downtrend)
            if supertrend[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Supertrend flips below price (uptrend)
            if supertrend[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Supertrend flip with volume confirmation AND daily trend alignment
            if close[i] > supertrend[i] and daily_close[i] > daily_trend[i]:
                # Price above Supertrend AND daily close above daily EMA50 -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < supertrend[i] and daily_close[i] < daily_trend[i]:
                # Price below Supertrend AND daily close below daily EMA50 -> short
                position = -1
                signals[i] = -0.25
    
    return signals