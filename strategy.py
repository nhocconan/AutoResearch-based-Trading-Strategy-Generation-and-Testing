#!/usr/bin/env python3
name = "6h_Williams_Alligator_Trend_With_1w_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator lines (Jaw, Teeth, Lips) on 6h
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    # Jaw (blue line) - 13-period smoothed with 8-period shift
    jaw_raw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw_raw, jaw_shift)
    jaw[:jaw_shift] = np.nan
    
    # Teeth (red line) - 8-period smoothed with 5-period shift
    teeth_raw = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth_raw, teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips (green line) - 5-period smoothed with 3-period shift
    lips_raw = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips_raw, lips_shift)
    lips[:lips_shift] = np.nan
    
    # Load 1-week data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA(20) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Alligator alignment conditions (no look-ahead)
    # Bullish alignment: Lips > Teeth > Jaw (green above red above blue)
    # Bearish alignment: Jaw > Teeth > Lips (blue above red above green)
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (jaw > teeth) & (teeth > lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_shift, teeth_shift, lips_shift)  # Ensure all lines available
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment + price above 1w EMA (uptrend filter)
            if bullish_align[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price below 1w EMA (downtrend filter)
            elif bearish_align[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish Alligator alignment or price below 1w EMA
            if bearish_align[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish Alligator alignment or price above 1w EMA
            if bullish_align[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator on 6h with 1-week EMA(20) trend filter.
# The Alligator identifies trend phases: when lines are intertwined (sleeping), no trend;
# when they diverge in order (Lips > Teeth > Jaw for uptrend, Jaw > Teeth > Lips for downtrend),
# a strong trend is present. The 1w EMA filter ensures alignment with the higher-timeframe
# trend, preventing counter-trend entries. This combination works in both bull and bear markets
# by only taking trades in the direction of the weekly trend. Position size 0.25 limits drawdown
# during choppy periods when the Alligator lines are intertwined (no signals). Target: 50-150
# total trades over 4 years (12-37/year) to avoid fee drag while capturing strong trends.