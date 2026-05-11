#!/usr/bin/env python3
"""
12h_Williams_Alligator_Rebound
Hypothesis: Price rebounds off Alligator's teeth (middle line) on 12h with 1d volume spike and 1w trend filter. Works in bull/bear as mean-reversion within trend.
Target: 15-30 trades/year to avoid fee drag.
"""

name = "12h_Williams_Alligator_Rebound"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close):
    """Returns (jaw, teeth, lips) lines."""
    # Jaw: Blue line (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: Red line (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: Green line (5-period SMMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # Williams Alligator on 12h
    jaw, teeth, lips = williams_alligator(high_12h, low_12h, close_12h)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # same timeframe
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # 1d volume spike filter (above 1.5x median of last 20 days)
    vol_1d = df_1d['volume'].values
    vol_median_1d = pd.Series(vol_1d).rolling(window=20, min_periods=10).median().values
    vol_threshold_1d = vol_median_1d * 1.5
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold_1d)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for Alligator and EMA
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                # Check stoploss via price crossing jaws
                if position == 1 and close_12h[i] < jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] > jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume condition: current 12h volume above 1d threshold
        vol_ok = volume_12h[i] > vol_1d_aligned[i]
        
        # Trend condition: price above/below 1w EMA50
        trend_up = close_12h[i] > ema50_1w_aligned[i]
        trend_down = close_12h[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price touches teeth from below, lips above teeth, volume spike, uptrend
            if (close_12h[i] >= teeth_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            # Short: price touches teeth from above, lips below teeth, volume spike, downtrend
            elif (close_12h[i] <= teeth_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            if position == 1:
                # Exit: price crosses below jaw (stop) or lips cross below teeth (signal)
                if close_12h[i] < jaw_aligned[i] or lips_aligned[i] < teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price crosses above jaw (stop) or lips cross above teeth (signal)
                if close_12h[i] > jaw_aligned[i] or lips_aligned[i] > teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals