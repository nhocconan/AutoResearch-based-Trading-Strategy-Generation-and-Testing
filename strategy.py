#!/usr/bin/env python3
# 4H_1D_Alligator_ElderRay_Trend
# Hypothesis: On 4h timeframe, enter long when Williams Alligator lines are bullish (jaw > teeth > lips) and Elder Ray bull power > 0 with 1d uptrend confirmation.
# Short when Alligator lines are bearish (jaw < teeth < lips) and Elder Ray bear power < 0 with 1d downtrend.
# Uses 1d EMA(50) for trend filter to avoid counter-trend trades.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years).

name = "4H_1D_Alligator_ElderRay_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema_50_1d
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Williams Alligator on 1d: SMA of median price (HL/2)
    # Jaw: 13-period SMA, 8 bars ahead
    # Teeth: 8-period SMA, 5 bars ahead
    # Lips: 5-period SMA, 3 bars ahead
    median_price_1d = (high_1d + low_1d) / 2
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 1d indicators to 4h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish (jaw > teeth > lips) + Bull Power > 0 + 1d uptrend
            if (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                bull_power_aligned[i] > 0 and trend_up_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish (jaw < teeth < lips) + Bear Power < 0 + 1d downtrend
            elif (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and
                  bear_power_aligned[i] < 0 and not trend_up_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0
            if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]) or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power >= 0
            if not (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]) or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals