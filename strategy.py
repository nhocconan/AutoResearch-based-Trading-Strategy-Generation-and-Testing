#!/usr/bin/env python3
# 6h_Donchian_Breakout_WeeklyTrend_WeeklyVol
# Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high with price > weekly EMA40 and volume > 1.5x 20-period MA.
# Enter short when price breaks below 20-period Donchian low with price < weekly EMA40 and volume > 1.5x 20-period MA.
# Exit when price crosses back below Donchian midpoint (for longs) or above midpoint (for shorts).
# Uses weekly timeframe for trend filter and volume confirmation to avoid false breakouts in low volatility.
# Targets 15-30 trades/year for low fee drag and works in both bull and bear markets by following weekly trend.

name = "6h_Donchian_Breakout_WeeklyTrend_WeeklyVol"
timeframe = "6h"
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
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly EMA40 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema40 = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_ema40_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema40)
    
    # Calculate 20-period volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_ema40_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with price > weekly EMA40 and volume > 1.5x MA
            if close[i] > donchian_high[i] and close[i] > weekly_ema40_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with price < weekly EMA40 and volume > 1.5x MA
            elif close[i] < donchian_low[i] and close[i] < weekly_ema40_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals