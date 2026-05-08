#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 13-period EMA and market regime filter.
# Elder Ray measures bull/bear power relative to EMA13. In bull regime (EMA13 rising), go long on Bull Power > 0.
# In bear regime (EMA13 falling), go short on Bear Power < 0. Uses volume confirmation for signal quality.
# Designed for low trade frequency (15-30/year) to avoid fee drag. Works in both trending and ranging markets.

name = "6h_1dElderRay_EMA13_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate EMA13 slope for regime detection (rising/falling)
    ema13_slope_1d = np.diff(ema13_1d, prepend=ema13_1d[0])
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13_slope_aligned = align_htf_to_ltf(prices, df_1d, ema13_slope_1d)
    
    # Volume confirmation: 6h volume spike (1.5x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bull regime (EMA13 rising) + Bull Power > 0 + volume spike
            if ema13_slope_aligned[i] > 0 and bull_power_aligned[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bear regime (EMA13 falling) + Bear Power < 0 + volume spike
            elif ema13_slope_aligned[i] < 0 and bear_power_aligned[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative or regime changes to bear
            if bull_power_aligned[i] <= 0 or ema13_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive or regime changes to bull
            if bear_power_aligned[i] >= 0 or ema13_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals