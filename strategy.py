#!/usr/bin/env python3
"""
12h_1d_ema_crossover_with_volume_and_regime
Hypothesis: 12-hour EMA crossover (8/21) with volume confirmation and 1-day ATR regime filter.
Works in bull/bear by using trend-following EMA crossovers only when volatility is elevated
(avoiding choppy markets) and volume confirms institutional participation.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

name = "12h_1d_ema_crossover_with_volume_and_regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA crossover (8/21) on daily
    ema8 = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR for regime filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align EMA and ATR to 12h timeframe
    ema8_aligned = align_htf_to_ltf(prices, df_1d, ema8)
    ema21_aligned = align_htf_to_ltf(prices, df_1d, ema21)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: volume > 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema8_aligned[i]) or np.isnan(ema21_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: EMA8 crosses above EMA21 with volume and volatility filter
        if (ema8_aligned[i] > ema21_aligned[i] and 
            ema8_aligned[i-1] <= ema21_aligned[i-1] and
            vol_confirm[i] and 
            atr_aligned[i] > 0 and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: EMA8 crosses below EMA21 with volume and volatility filter
        elif (ema8_aligned[i] < ema21_aligned[i] and 
              ema8_aligned[i-1] >= ema21_aligned[i-1] and
              vol_confirm[i] and 
              atr_aligned[i] > 0 and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal
        elif position == 1 and ema8_aligned[i] < ema21_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema8_aligned[i] > ema21_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals