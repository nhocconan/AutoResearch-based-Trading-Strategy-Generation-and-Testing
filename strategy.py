#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from daily timeframe with trend filter and volume spike confirmation.
Enters long when price breaks above R1 level with daily uptrend and volume above average.
Enters short when price breaks below S1 level with daily downtrend and volume above average.
Uses daily pivot levels for structure and filters to reduce false breakouts in ranging markets.
Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4"""
    # Pivot Point (PP)
    pp = (high + low + close) / 3
    # Range
    range_hl = high - low
    
    # Resistance levels
    r1 = pp + (range_hl * 1.0833 / 2)
    r2 = pp + (range_hl * 1.0833 / 2 * 2)
    r3 = pp + (range_hl * 1.0833 / 2 * 3)
    r4 = pp + (range_hl * 1.0833 / 2 * 4)
    
    # Support levels
    s1 = pp - (range_hl * 1.0833 / 2)
    s2 = pp - (range_hl * 1.0833 / 2 * 2)
    s3 = pp - (range_hl * 1.0833 / 2 * 3)
    s4 = pp - (range_hl * 1.0833 / 2 * 4)
    
    return r1, r2, r3, r4, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily Camarilla pivots
    r1_1d, r2_1d, r3_1d, r4_1d, pp_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume average for volume spike confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        # Get aligned values for current 4h bar
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma20_aligned[i]
        vol_val = volume[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(ema50_val) or np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with daily uptrend and volume spike
            if (close[i] > r1_val and 
                close[i] > ema50_val and 
                vol_val > vol_ma_val * 1.5):  # 50% above average volume
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with daily downtrend and volume spike
            elif (close[i] < s1_val and 
                  close[i] < ema50_val and 
                  vol_val > vol_ma_val * 1.5):  # 50% above average volume
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend turns down
            if (close[i] < r1_val or close[i] < ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend turns up
            if (close[i] > s1_val or close[i] > ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals