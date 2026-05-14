#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + 1d Elder Ray combination. 
# Long when Alligator is bullish (jaw < teeth < lips) AND 1d Bear Power > 0 (bulls in control).
# Short when Alligator is bearish (jaw > teeth > lips) AND 1d Bull Power < 0 (bears in control).
# Exit on Alligator cross reversal. Uses smoothed medians to reduce whipsaw.
# Designed for 6h timeframe: low frequency (~12-25 trades/year), works in both bull and bear markets
# by combining trend (Alligator) with market power (Elder Ray) filters.

name = "6h_WilliamsAlligator_1dElderRay_v1"
timeframe = "6h"
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
    
    # --- 6h Williams Alligator (smoothed medians) ---
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # --- 1d Elder Ray (Bull/Bear Power) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA of close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = high - EMA(13)
    bull_power_1d = high_1d - ema_13_1d
    # Bear Power = low - EMA(13)
    bear_power_1d = low_1d - ema_13_1d
    
    # Align to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator bullish (jaw < teeth < lips) AND Bear Power > 0 (bulls in control)
            if (jaw[i] < teeth[i] < lips[i]) and (bear_power_1d_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish (jaw > teeth > lips) AND Bull Power < 0 (bears in control)
            elif (jaw[i] > teeth[i] > lips[i]) and (bull_power_1d_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish (jaw > teeth)
            if jaw[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish (jaw < teeth)
            if jaw[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals