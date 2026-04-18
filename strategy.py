#!/usr/bin/env python3
"""
12h_1w_Pivot_R1S1_Breakout_With_TrendAndVolume
Hypothesis: 12h price breaks above/below weekly Camarilla R1/S1 levels with volume spike and 1d trend confirmation.
Weekly pivots provide strong support/resistance levels that work in both bull and bear markets.
Volume confirms momentum, 1d EMA34 filter avoids counter-trend trades.
Designed for 15-25 trades/year to minimize fee drag while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla pivot points: calculated from previous week's OHLC
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    
    # Align weekly levels to 12h timeframe
    pivot_1w_aligned = align_ltf_to_htf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_ltf_to_htf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_ltf_to_htf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_ltf_to_htf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_ltf_to_htf(prices, df_1w, s2_1w)
    
    # Volume spike: >1.5x 30-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Trend filter: 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 30)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below pivot OR trend turns down
            if price < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above pivot OR trend turns up
            if price > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1w_Pivot_R1S1_Breakout_With_TrendAndVolume"
timeframe = "12h"
leverage = 1.0