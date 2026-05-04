#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide high-probability support/resistance from prior day
# Breakout above R1 with 1d uptrend and volume confirmation = long entry
# Breakdown below S1 with 1d downtrend and volume confirmation = short entry
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# Discrete sizing 0.25 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + (rng * 1.1 / 12)
    camarilla_s1 = close_1d - (rng * 1.1 / 12)
    
    # Shift to use only prior completed 1d bar (no look-ahead)
    camarilla_r1_shifted = np.roll(camarilla_r1, 1)
    camarilla_s1_shifted = np.roll(camarilla_s1, 1)
    camarilla_r1_shifted[0] = np.nan
    camarilla_s1_shifted[0] = np.nan
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_shifted)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R1 AND 1d EMA34 uptrend AND volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below S1 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 OR 1d EMA34 turns downtrend
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 OR 1d EMA34 turns uptrend
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals