#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_1dTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull/Bear Power) breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period median). 
Enters long when Bull Power > 0 and price closes above prior 6h high with volume confirmation and bullish 1d trend. 
Enters short when Bear Power < 0 and price closes below prior 6h low with volume confirmation and bearish 1d trend. 
Exits on reversal of Elder Ray signal. Uses discrete position sizing (0.25) to minimize churn. 
Target: 50-150 trades over 4 years. Works in both bull and bear markets by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary 6h)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 13-period EMA, 20-period volume median, 34-period EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or np.isnan(vol_median[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Bull Power > 0 + close above prior high + volume confirm + bullish 1d trend
        if bull_power[i] > 0 and close[i] > high[i-1] and volume_confirm[i] and close[i] > ema34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Bear Power < 0 + close below prior low + volume confirm + bearish 1d trend
        elif bear_power[i] < 0 and close[i] < low[i-1] and volume_confirm[i] and close[i] < ema34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: reversal of Elder Ray signal (long exits when Bear Power >= 0, short exits when Bull Power <= 0)
        elif position == 1 and bear_power[i] >= 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bull_power[i] <= 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ElderRay_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0