#!/usr/bin/env python3
"""
4h_Keltner_Momentum_Trend
Hypothesis: Uses Keltner Channel (ATR-based) breakouts with EMA trend filter on 4h. 
Enters long when price breaks above upper band with EMA21 > EMA50, short when breaks below lower band with EMA21 < EMA50.
Volume confirmation ensures momentum. Designed for fewer trades (~20-40/year) with strong trend capture in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    atr_period = 10
    ema_period = 20
    keltner_mult = 2.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate EMA for Keltner middle line
    ema_mid = np.full(n, np.nan)
    k = 2 / (ema_period + 1)
    for i in range(ema_period, n):
        if i == ema_period:
            ema_mid[i] = np.mean(close[i-ema_period+1:i+1])
        else:
            ema_mid[i] = close[i] * k + ema_mid[i-1] * (1 - k)
    
    # Upper and lower bands
    upper = ema_mid + keltner_mult * atr
    lower = ema_mid - keltner_mult * atr
    
    # EMA trend filter (21 and 50)
    ema21 = np.full(n, np.nan)
    ema50 = np.full(n, np.nan)
    k21 = 2 / (21 + 1)
    k50 = 2 / (50 + 1)
    for i in range(50, n):
        if i == 50:
            ema21[i] = np.mean(close[i-21+1:i+1]) if i >= 21 else np.nan
            ema50[i] = np.mean(close[i-50+1:i+1])
        else:
            if not np.isnan(ema21[i-1]):
                ema21[i] = close[i] * k21 + ema21[i-1] * (1 - k21)
            if not np.isnan(ema50[i-1]):
                ema50[i] = close[i] * k50 + ema50[i-1] * (1 - k50)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with uptrend and volume spike
            if close[i] > upper[i] and ema21[i] > ema50[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with downtrend and volume spike
            elif close[i] < lower[i] and ema21[i] < ema50[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below EMA21 or trend weakens
            if close[i] < ema21[i] or ema21[i] <= ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above EMA21 or trend weakens
            if close[i] > ema21[i] or ema21[i] >= ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Momentum_Trend"
timeframe = "4h"
leverage = 1.0