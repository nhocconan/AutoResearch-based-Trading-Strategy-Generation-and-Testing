#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + Williams Alligator confluence with 12h trend filter.
- Primary timeframe: 6h for lower trade frequency and reduced fee drag.
- HTF: 12h for trend direction (bullish if EMA13 > EMA34, bearish if EMA13 < EMA34).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Williams Alligator: Jaw (EMA13), Teeth (EMA8), Lips (EMA5) on 6h.
- Entry: Long when Bull Power > 0 AND Lips > Teeth > Jaw (bullish alignment) AND 12h trend bullish.
         Short when Bear Power < 0 AND Lips < Teeth < Jaw (bearish alignment) AND 12h trend bearish.
- Exit: Opposite Alligator alignment (Lips crosses Teeth) or Elder Power divergence.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy combines trend-following (Alligator) with momentum (Elder Ray) and HTF trend filter
to capture sustained moves while avoiding whipsaws. Works in both bull and bear markets by
only taking trades in the direction of the 12h trend, with Elder Ray confirming momentum
and Alligator providing dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA13 and EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h Elder Ray components (EMA13)
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Calculate 6h Williams Alligator
    ema5_6h = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values   # Lips
    ema8_6h = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values   # Teeth
    ema13_6h_jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw
    
    # Align HTF indicators to 6h
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    ema5_6h_aligned = align_htf_to_ltf(prices, df_12h, ema5_6h)
    ema8_6h_aligned = align_htf_to_ltf(prices, df_12h, ema8_6h)
    ema13_6h_jaw_aligned = align_htf_to_ltf(prices, df_12h, ema13_6h_jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 13, 8, 5)  # Need enough bars for all EMAs
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_12h_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema5_6h_aligned[i]) or np.isnan(ema8_6h_aligned[i]) or
            np.isnan(ema13_6h_jaw_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_bullish = ema13_12h_aligned[i] > ema34_12h_aligned[i]
        trend_bearish = ema13_12h_aligned[i] < ema34_12h_aligned[i]
        
        # Determine Alligator alignment
        lips_above_teeth = ema5_6h_aligned[i] > ema8_6h_aligned[i]
        teeth_above_jaw = ema8_6h_aligned[i] > ema13_6h_jaw_aligned[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = ema5_6h_aligned[i] < ema8_6h_aligned[i]
        teeth_below_jaw = ema8_6h_aligned[i] < ema13_6h_jaw_aligned[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Check for entry signals
            if bull_power_aligned[i] > 0 and bullish_alignment and trend_bullish:
                signals[i] = 0.25
                position = 1
            elif bear_power_aligned[i] < 0 and bearish_alignment and trend_bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish Alligator alignment OR Bear Power turns negative
            if not bullish_alignment or bear_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Alligator alignment OR Bull Power turns positive
            if not bearish_alignment or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_12hEMATrend_v1"
timeframe = "6h"
leverage = 1.0