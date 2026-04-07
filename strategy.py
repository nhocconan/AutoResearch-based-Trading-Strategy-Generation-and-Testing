#!/usr/bin/env python3
"""
1h_volume_bull_bear_crossover_v1
Hypothesis: 1h EMA(9/21) crossover with volume confirmation and 4h trend filter.
Go long when fast EMA crosses above slow EMA, volume > 20-period average, and 4h EMA(50) is rising.
Go short when fast EMA crosses below slow EMA, volume > 20-period average, and 4h EMA(50) is falling.
Uses volume filter to ensure momentum behind moves and 4h trend to avoid counter-trend trades.
Designed for 15-35 trades/year per symbol to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_bull_bear_crossover_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA: fast 9, slow 21
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h EMA50 slope (rising/falling)
    ema50_slope = np.diff(ema50_4h_aligned, prepend=ema50_4h_aligned[0])
    ema50_rising = ema50_slope > 0
    ema50_falling = ema50_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if data not available
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        if position == 1:  # Long position
            # Exit: EMA cross down or 4h EMA50 falling
            if ema_cross_down or ema50_falling[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA cross up or 4h EMA50 rising
            if ema_cross_up or ema50_rising[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: EMA cross up with volume confirmation and 4h EMA50 rising
            if ema_cross_up and vol_confirmed and ema50_rising[i]:
                position = 1
                signals[i] = 0.20
            # Short: EMA cross down with volume confirmation and 4h EMA50 falling
            elif ema_cross_down and vol_confirmed and ema50_falling[i]:
                position = -1
                signals[i] = -0.20
    
    return signals