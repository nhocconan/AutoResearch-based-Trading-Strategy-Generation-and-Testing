#!/usr/bin/env python3
# 1d_Alligator_WeeklyTrend
# Hypothesis: Use Williams Alligator (jaw/teeth/lips) on daily chart for trend direction.
# Enter long when price closes above lips (green) with price > jaw (blue) and teeth (red),
# enter short when price closes below lips with price < jaw and teeth.
# Exit when price crosses back into the Alligator's mouth (between jaw and teeth).
# Weekly trend filter (SMA50) ensures we only trade in higher timeframe trend direction.
# Designed for low frequency (10-20 trades/year) by using daily timeframe with weekly filter.
# Works in both bull and bear markets by following higher timeframe trend and using Alligator's convergence/divergence.

name = "1d_Alligator_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Daily data for Williams Alligator ===
    # Williams Alligator: SMAs shifted into the future
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward  
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    # We calculate SMMA (smoothed moving average) as EMA with alpha = 1/period
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)  # Shift forward 8 bars
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)  # Shift forward 5 bars
    teeth[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips, 3)  # Shift forward 3 bars
    lips[:3] = np.nan  # First 3 values invalid after shift
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator conditions
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        if position == 0:
            # LONG: Lips above teeth > jaw (Alligator eating up) AND weekly uptrend
            if lips_above_teeth and teeth_above_jaw and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips below teeth < jaw (Alligator eating down) AND weekly downtrend
            elif lips_below_teeth and teeth_below_jaw and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses back into Alligator's mouth (between jaw and teeth)
            # or weekly trend reverses
            if (lips[i] <= teeth[i] and lips[i] >= jaw[i]) or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back into Alligator's mouth (between jaw and teeth)
            # or weekly trend reverses
            if (lips[i] >= teeth[i] and lips[i] <= jaw[i]) or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals