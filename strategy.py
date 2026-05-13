#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3_S3_Breakout_With_Volume_Confirmation
# Hypothesis: On 12h timeframe, breakout of Camarilla R3/S3 levels (from prior day) 
# with volume confirmation captures institutional breakouts. 
# In bull markets, price breaks above R3 and continues upward; 
# in bear markets, breaks below S3 and continues downward. 
# The Camarilla levels act as natural support/resistance from prior day's range.
# Volume spike confirms breakout validity. Works in both regimes by following breakout direction.

name = "12h_Camarilla_Pivot_R3_S3_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla calculation (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 as breakout levels
    prior_close = df_1d['close'].values
    prior_high = df_1d['high'].values
    prior_low = df_1d['low'].values
    prior_range = prior_high - prior_low
    
    camarilla_r3 = prior_close + 1.1 * prior_range
    camarilla_s3 = prior_close - 1.1 * prior_range
    
    # Align Camarilla levels to 12h timeframe (available after prior day close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: volume > 2.0 * 20-period average (~10 days at 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 with volume spike
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with volume spike
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below Camarilla R3
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above Camarilla S3
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals