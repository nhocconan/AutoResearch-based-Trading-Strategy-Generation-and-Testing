#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d trend filter.
# Long when: Green phase (jaw<teeth<lips), Bull Power > 0, price > 1d EMA50.
# Short when: Red phase (lips<teeth<jaw), Bear Power < 0, price < 1d EMA50.
# Exit when: Alligator lines cross in opposite direction or power signal reverses.
# Uses Williams Alligator (13,8,5 SMAs) for trend phase and Elder Ray (Bull/Bear Power) for strength.
# 1d EMA50 filter ensures alignment with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: 13,8,5 period SMAs (median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Green phase (jaw < teeth < lips), Bull Power > 0, price > 1d EMA50
        if (jaw[i] < teeth[i] and teeth[i] < lips[i] and
            bull_power[i] > 0 and
            close[i] > ema50_1d_aligned[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: Red phase (lips < teeth < jaw), Bear Power < 0, price < 1d EMA50
        elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
              bear_power[i] < 0 and
              close[i] < ema50_1d_aligned[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: Alligator crosses opposite direction or power signal reverses
        elif position == 1 and (jaw[i] > teeth[i] or bull_power[i] <= 0):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (teeth[i] > jaw[i] or bear_power[i] >= 0):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1dEMA50"
timeframe = "6h"
leverage = 1.0