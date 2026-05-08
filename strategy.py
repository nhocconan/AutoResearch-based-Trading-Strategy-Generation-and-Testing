#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination for trend identification with Elder Ray for momentum confirmation.
# Long when: Alligator jaw < teeth < lips (bullish alignment) AND Elder Ray Bull Power > 0 AND price > EMA13.
# Short when: Alligator jaw > teeth > lips (bearish alignment) AND Elder Ray Bear Power < 0 AND price < EMA13.
# Exit when Alligator lines re-cross (jaw crosses teeth) indicating trend exhaustion.
# Alligator identifies trend structure, Elder Ray confirms momentum behind the move.
# Works in both bull and bear markets by following the trend defined by Alligator alignment.

name = "6h_Alligator_ElderRay_EMA13"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5).values   # Red line
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = df_1d['close'].ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = (df_1d['high'] - ema13).values
    bear_power = (df_1d['low'] - ema13).values
    
    # Align Alligator and Elder Ray to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h EMA13 for entry filter
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 13)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema13_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: jaw < teeth < lips
            bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
            # Bearish Alligator alignment: jaw > teeth > lips
            bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
            
            # Long conditions: bullish alignment + positive Bull Power + price > EMA13
            long_cond = bullish_alignment and (bull_power_aligned[i] > 0) and (close[i] > ema13_6h[i])
            # Short conditions: bearish alignment + negative Bear Power + price < EMA13
            short_cond = bearish_alignment and (bear_power_aligned[i] < 0) and (close[i] < ema13_6h[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator re-cross (jaw crosses above teeth) indicating trend weakening
            if jaw_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator re-cross (jaw crosses below teeth) indicating trend weakening
            if jaw_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals