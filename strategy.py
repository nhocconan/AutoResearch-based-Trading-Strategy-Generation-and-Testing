#!/usr/bin/env python3
# 4h_WilliamsAlligator_12hTrend
# Hypothesis: Williams Alligator (3 SMAs: Jaw 13, Teeth 8, Lips 5) identifies market phases.
# Long when Lips > Teeth > Jaw (bullish alignment) and 12h close > 12h EMA50; short when Lips < Teeth < Jaw (bearish alignment) and 12h close < 12h EMA50.
# Exit when Alligator lines cross (Lips crosses Teeth) or trend weakens. Uses 12h trend filter to avoid whipsaw.
# Works in trending markets (bull/bear) and avoids range-bound conditions via trend filter.

name = "4h_WilliamsAlligator_12hTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Alligator components on 4h data
    # Jaw: 13-period SMMA (smoothed median)
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    # SMMA is similar to smoothed moving average, approximated by EMA for simplicity
    jaw = pd.Series(high).ewm(alpha=1/13, adjust=False).mean().values  # Approximation
    teeth = pd.Series(high).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(high).ewm(alpha=1/5, adjust=False).mean().values
    
    # Alternative: use median of high-low-close for more robustness
    typical_price = (high + low + close) / 3
    jaw = pd.Series(typical_price).ewm(alpha=1/13, adjust=False).mean().values
    teeth = pd.Series(typical_price).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(typical_price).ewm(alpha=1/5, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 12h close for trend filter
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        close_12h_current = close_12h_aligned[i]
        
        trend_up = close_12h_current > ema50_12h_aligned[i]
        trend_down = close_12h_current < ema50_12h_aligned[i]
        
        # Alligator alignment checks
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # LONG: bullish Alligator alignment AND 12h uptrend
            if bullish_alignment and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish Alligator alignment AND 12h downtrend
            elif bearish_alignment and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (Lips crosses below Teeth) OR trend weakens
            if not bullish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (Lips crosses above Teeth) OR trend weakens
            if not bearish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals