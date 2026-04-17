#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_Volume_Momentum_v1
Breakout at Camarilla R1/S1 with volume confirmation and momentum filter (ROC).
Uses 12h trend filter: price above/below 12h EMA34.
Exit when price crosses back below/above R1/S1 or momentum weakens.
Designed to capture breakouts with institutional volume and trend alignment.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Camarilla Pivot Levels from previous day ===
    # Calculate using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First bar: use current values as fallback
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r4 = pivot + (range_ * 1.1 / 2)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # === ROC(10) for momentum ===
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 12h EMA34 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(roc[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, positive momentum, above 12h EMA34
            if (close[i] > r1[i] and 
                vol_ratio[i] > 1.5 and 
                roc[i] > 0 and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume spike, negative momentum, below 12h EMA34
            elif (close[i] < s1[i] and 
                  vol_ratio[i] > 1.5 and 
                  roc[i] < 0 and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses back below R1 OR momentum turns negative
            if (close[i] < r1[i] or 
                roc[i] < 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above S1 OR momentum turns positive
            if (close[i] > s1[i] or 
                roc[i] > 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_Momentum_v1"
timeframe = "6h"
leverage = 1.0