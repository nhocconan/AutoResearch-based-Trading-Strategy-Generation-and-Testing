#!/usr/bin/env python3
# 6H_ELDER_RAY_POWER_1D_TREND_VOLUME_FILTER
# Hypothesis: On 6h timeframe, use daily Elder Ray (Bull/Bear Power) combined with 1d EMA trend filter and volume confirmation.
# Elder Ray measures bull/bear strength by comparing daily high/low to EMA.
# Enter long when Bull Power > 0 (bulls in control) AND price above 1d EMA34 (uptrend) AND volume spike.
# Enter short when Bear Power < 0 (bears in control) AND price below 1d EMA34 (downtrend) AND volume spike.
# Exit when Bull/Bear Power crosses zero (loss of momentum).
# Designed to work in both bull and bear markets via trend alignment and momentum filtering.

name = "6H_ELDER_RAY_POWER_1D_TREND_VOLUME_FILTER"
timeframe = "6h"
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
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA, Bear Power = Low - EMA
    # Use the same EMA34 for consistency
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema34
    bear_power = low_1d - ema34
    
    # Align all daily indicators to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bulls in control) AND price above EMA (uptrend) AND volume spike
            if bull_power_aligned[i] > 0 and close[i] > ema34_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bears in control) AND price below EMA (downtrend) AND volume spike
            elif bear_power_aligned[i] < 0 and close[i] < ema34_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative (loss of bullish momentum)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive (loss of bearish momentum)
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals