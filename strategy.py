#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when Bull Power < 0 AND Bear Power > 0 AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Elder Ray signals reverse (Bull Power crosses zero or Bear Power crosses zero).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 30-60 trades/year per symbol.
Elder Ray measures bull/bear strength via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
The daily EMA50 provides trend alignment to avoid counter-trend entries in choppy markets.
Volume confirmation reduces false signals. Works in both bull and bear markets by following the 1d trend.
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
    
    # Load 6h data for Elder Ray calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 on 6h data for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND close < 1d EMA50 AND volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray signals reverse
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power crosses below zero OR Bear Power crosses above zero
                if bull_power[i] <= 0 or bear_power[i] >= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bull Power crosses above zero OR Bear Power crosses below zero
                if bull_power[i] >= 0 or bear_power[i] <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0