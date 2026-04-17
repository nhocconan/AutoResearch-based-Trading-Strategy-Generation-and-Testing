#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_Crossover
Strategy: TRIX(12) crossover with volume spike on 12h, filtered by 1d EMA34 trend.
Long: TRIX crosses above 0 + volume > 1.5x average + 1d EMA34 rising
Short: TRIX crosses below 0 + volume > 1.5x average + 1d EMA34 falling
Exit: Opposite TRIX cross or trend reversal
Position size: 0.25
Designed to capture momentum shifts with volume confirmation in trending markets.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(12) on 12h
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    trix = trix.fillna(0).values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20)  # TRIX needs 3x EMA + volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: 1d EMA34 rising/falling
        ema34_rising = i > 0 and ema34_1d_aligned[i] > ema34_1d_aligned[i-1]
        ema34_falling = i > 0 and ema34_1d_aligned[i] < ema34_1d_aligned[i-1]
        
        # TRIX crossover signals
        trix_cross_up = i > 0 and trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = i > 0 and trix[i-1] >= 0 and trix[i] < 0
        
        if position == 0:
            # Long: TRIX crosses up + rising trend + volume spike
            if trix_cross_up and ema34_rising and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses down + falling trend + volume spike
            elif trix_cross_down and ema34_falling and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses down or trend turns down
            if trix_cross_down or not ema34_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses up or trend turns up
            if trix_cross_up or not ema34_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_Crossover"
timeframe = "12h"
leverage = 1.0