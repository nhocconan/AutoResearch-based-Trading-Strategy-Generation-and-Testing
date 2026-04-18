#!/usr/bin/env python3
"""
12h Pivot Range Breakout with Volume Confirmation and 1d Trend Filter
Hypothesis: Price breaking above/below the previous day's high/low (pivot range) 
with volume confirmation and aligned with 1d EMA trend captures momentum moves.
Works in both bull and bear markets by filtering counter-trend trades using 1d EMA.
Target: 15-25 trades/year to minimize fee decay while capturing strong directional moves.
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
    
    # Get 1d data for trend filter and pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous day's high and low for pivot range
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align pivot levels to 12h timeframe (wait for 1d bar to close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume filter: current volume > 2.0x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema34_1d_aligned[i]
        ph = prev_high_aligned[i]
        pl = prev_low_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above previous day's high with volume, in uptrend
            if price > ph and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low with volume, in downtrend
            elif price < pl and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns below previous day's high or trend weakens
            if price < ph or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns above previous day's low or trend weakens
            if price > pl or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PivotRange_Breakout_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0