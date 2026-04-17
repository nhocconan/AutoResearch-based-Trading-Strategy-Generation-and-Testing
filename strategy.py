#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray combination for trend detection.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
- Elder Ray: Bull Power = High - EMA13, Bear Power = Lows - EMA13
- Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND Bear Power rising
- Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND Bear Power < 0 AND Bull Power falling
- Uses 6h primary timeframe with 1d HTF for Alligator alignment filter (more stable)
- Fixed position size 0.25 to manage drawdown
- Designed for low trade frequency (20-60/year) with trend-following edge in both bull/bear regimes
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
    
    # Get 1d data for Williams Alligator and EMA13 (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate median price for Alligator
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Williams Alligator lines (SMAs on median price)
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # 13-period, 8-shift
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values   # 8-period, 5-shift
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values    # 5-period, 3-shift
    
    # Apply shifts (Alligator definition)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Set initial invalid values to NaN
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align all HTF indicators to primary timeframe (6h)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish_aligned = lips_val > teeth_val > jaw_val
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish_aligned = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish Alligator + positive Bull Power + rising Bear Power (less negative)
            if bullish_aligned and bull_power > 0 and (i == start_idx or bear_power > bear_power_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + negative Bear Power + falling Bull Power (less positive)
            elif bearish_aligned and bear_power < 0 and (i == start_idx or bull_power < bull_power_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit on Alligator bearish alignment or Bull Power turning negative
            if not bullish_aligned or bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on Alligator bullish alignment or Bear Power turning positive
            if not bearish_aligned or bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_Trend"
timeframe = "6h"
leverage = 1.0