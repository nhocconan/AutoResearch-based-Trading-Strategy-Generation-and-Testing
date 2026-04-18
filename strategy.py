#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: In both bull and bear markets, price breaking out of Donchian channels
with volume confirmation and higher timeframe trend alignment captures sustained
momentum moves. The 1d EMA50 filter prevents counter-trend trades, while volume
confirms institutional participation. Targeting 15-25 trades/year to minimize
fee drag while capturing significant moves. Breakouts from 20-period channels
identify meaningful support/resistance breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema50_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with volume, in uptrend
            if price > high_max[i] and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume, in downtrend
            elif price < low_min[i] and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to midpoint of channel or trend weakens
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if price < midpoint or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to midpoint of channel or trend weakens
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if price > midpoint or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0