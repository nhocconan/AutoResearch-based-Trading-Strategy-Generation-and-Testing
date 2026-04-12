#!/usr/bin/env python3
"""
6h_1d_Adaptive_Keltner_Channel_v1
Hypothesis: 6h timeframe with 1d adaptive Keltner channels (ATR-based) and volume confirmation.
Uses dynamic channel width based on volatility regime to capture breakouts in both trending and ranging markets.
Volatility regime filter prevents whipsaws in low volatility periods. Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Adaptive_Keltner_Channel_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR (14 period) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA (20 period) for middle line
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate adaptive multiplier based on volatility regime
    # Use 50-period ATR percentile to determine volatility regime
    atr_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma  # Current ATR relative to average
    # Adaptive multiplier: 1.5 in low vol, 2.5 in high vol
    mult = 1.5 + (atr_ratio - 1.0)  # Scale linearly around 1.0
    mult = np.clip(mult, 1.5, 2.5)  # Clamp between 1.5 and 2.5
    
    # Calculate Keltner channels
    upper = ema_1d + mult * atr_1d
    lower = ema_1d - mult * atr_1d
    
    # Align to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_1d, upper)
    lower_6h = align_htf_to_ltf(prices, df_1d, lower)
    ema_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20 period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.3x average
        volume_spike = volume[i] > vol_ma[i] * 1.3
        
        # Breakout conditions
        breakout_up = close[i] > upper_6h[i]
        breakout_down = close[i] < lower_6h[i]
        
        # Entry conditions: breakout with volume confirmation
        long_entry = breakout_up and volume_spike
        short_entry = breakout_down and volume_spike
        
        # Exit conditions: return to middle line or opposite band touch
        long_exit = close[i] < ema_6h[i] or close[i] > upper_6h[i] * 1.02
        short_exit = close[i] > ema_6h[i] or close[i] < lower_6h[i] * 0.98
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals