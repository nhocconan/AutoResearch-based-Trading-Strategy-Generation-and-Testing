#!/usr/bin/env python3
"""
6h_alligator_elderray_v1
Hypothesis: Combine Williams Alligator (trend detector) with Elder Ray (bull/bear power) on 6h timeframe.
- Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs. Trend when Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear).
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13). 
- Entry: Bull Power > 0 and Bear Power < 0 with Alligator bullish alignment → long.
         Bear Power < 0 and Bull Power > 0 with Alligator bearish alignment → short.
- Exit: When Alligator alignment breaks (Lips crosses Teeth).
- Use 1w trend filter: only trade in direction of weekly EMA(40) to avoid counter-trend trades.
- Designed for 15-30 trades/year on 6h to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_alligator_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # SMA(13,8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # SMA(8,5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values   # SMA(5,3)
    
    # Elder Ray components on 6h
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Alligator alignment signals
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    alligator_bullish = lips_above_teeth & teeth_above_jaw
    alligator_bearish = lips_below_teeth & teeth_below_jaw
    
    # Weekly trend filter (1w EMA40)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, min_periods=40, adjust=False).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Weekly trend: price above/below EMA40
    weekly_uptrend = close > ema40_1w_aligned
    weekly_downtrend = close < ema40_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema40_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator bullish alignment breaks
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator bearish alignment breaks
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Bull Power > 0, Bear Power < 0, Alligator bullish, weekly uptrend
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                alligator_bullish[i] and weekly_uptrend[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power < 0, Bull Power > 0, Alligator bearish, weekly downtrend
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  alligator_bearish[i] and weekly_downtrend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals