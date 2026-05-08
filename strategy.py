#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray and volume spike confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND 12h volume > 1.8x 24-period average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND 12h volume > 1.8x 24-period average.
# Exit when Alligator alignment breaks or Elder Ray signal reverses.
# Uses Williams Alligator (13,8,5 SMAs) for trend, Elder Ray (13 EMA) for power, volume for confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "12h_WilliamsAlligator_1dElderRay_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h: SMAs of median price
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean()
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean()
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean()
    jaws = jaws.values
    teeth = teeth.values
    lips = lips.values
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 12h
    bull_power_12h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_12h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 12h volume filter: current volume > 1.8x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_12h[i]) or np.isnan(bear_power_12h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish Alligator alignment, positive bull power, volume spike
            long_cond = (jaws[i] < teeth[i]) and (teeth[i] < lips[i]) and (bull_power_12h[i] > 0) and volume_filter[i]
            # Short conditions: Bearish Alligator alignment, negative bear power, volume spike
            short_cond = (jaws[i] > teeth[i]) and (teeth[i] > lips[i]) and (bear_power_12h[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or bull power turns negative
            if not ((jaws[i] < teeth[i]) and (teeth[i] < lips[i])) or (bull_power_12h[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or bear power turns positive
            if not ((jaws[i] > teeth[i]) and (teeth[i] > lips[i])) or (bear_power_12h[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals