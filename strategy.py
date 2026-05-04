#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume spike confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend direction and strength
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# Long: Alligator bullish (LIPS > TEETH > JAW) AND Bull Power > 0 AND volume spike
# Short: Alligator bearish (LIPS < TEETH < JAW) AND Bear Power > 0 AND volume spike
# Uses 1d timeframe for Alligator/Elder Ray calculation to reduce noise
# Volume confirmation (2.0x 20-period EMA) filters weak breakouts
# Designed for low trade frequency (12-37/year) to minimize fee drag
# Works in both bull and bear markets due to trend-following nature with volume filter

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for Williams Alligator and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator: SMAs of median price
    # Median price = (High + Low) / 2
    median_price_1d = (high_1d + low_1d) / 2
    
    # JAW: 13-period SMMA (smoothed moving average) of median price
    jaw_1d = pd.Series(median_price_1d).ewm(alpha=1/13, adjust=False).mean().values
    # TEETH: 8-period SMMA of median price
    teeth_1d = pd.Series(median_price_1d).ewm(alpha=1/8, adjust=False).mean().values
    # LIPS: 5-period SMMA of median price
    lips_1d = pd.Series(median_price_1d).ewm(alpha=1/5, adjust=False).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Using 13-period EMA of close as the reference (similar to Alligator's jaw)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align Alligator and Elder Ray components to 12h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND Bull Power > 0 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish AND Bear Power > 0 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  bear_power_aligned[i] > 0 and 
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0
            if (lips_aligned[i] <= teeth_aligned[i] or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power <= 0
            if (lips_aligned[i] >= teeth_aligned[i] or 
                bear_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals