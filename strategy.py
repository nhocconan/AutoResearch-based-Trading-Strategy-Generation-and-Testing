#!/usr/bin/env python3
"""
12h_WilliamsAlligator_JawTeethLips_Trend_1wTrend_VolumeFilter
Hypothesis: Williams Alligator (13,8,5 SMAs) defines trend direction (price > all = uptrend, price < all = downtrend). 
Trades only in direction of weekly trend (price vs weekly EMA50) with volume confirmation (>1.5x 20-bar avg). 
Uses 12h timeframe to target 50-150 total trades over 4 years. 
Williams Alligator avoids whipsaw by requiring alignment of all three lines. 
Weekly trend filter prevents counter-trend trades. Volume filter ensures momentum. 
Designed to work in both bull (follows uptrend) and bear (follows downtrend) markets.
"""

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
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # 1w trend filter: price vs EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # same timeframe
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, prices, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Alligator (13), EMA50 (50), volume avg (20)
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        # Alligator alignment: all three lines in order
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        uptrend_align = lips_above_teeth and teeth_above_jaw
        downtrend_align = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Determine weekly trend: price vs weekly EMA50
            weekly_uptrend = close_val > ema50
            weekly_downtrend = close_val < ema50
            
            if uptrend_align and weekly_uptrend and vol_conf:
                # Long when Alligator aligned up, weekly trend up, volume confirmed
                signals[i] = size
                position = 1
                entry_price = close_val
            elif downtrend_align and weekly_downtrend and vol_conf:
                # Short when Alligator aligned down, weekly trend down, volume confirmed
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: Alligator convergence (Lips crosses below Teeth) or weekly trend change
            if lips_val < teeth_val or close_val < ema50:  # Lips cross below Teeth or weekly trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: Alligator convergence (Lips crosses above Teeth) or weekly trend change
            if lips_val > teeth_val or close_val > ema50:  # Lips cross above Teeth or weekly trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_JawTeethLips_Trend_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0