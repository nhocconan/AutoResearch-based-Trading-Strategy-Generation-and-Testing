#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# Uses Williams Alligator (JAW/TEETH/LIPS) to determine trend direction,
# Elder Ray (Bull/Bear Power) for momentum confirmation, and volume spike
# for entry confirmation. Works in trending markets (bull/bear) and avoids
# ranging conditions. Designed for low trade frequency (~20-40/year) to
# minimize fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for Williams Alligator (13/8/5 SMAs of median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    median_price_1w = (df_1w['high'].values + df_1w['low'].values) / 2
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values  # SMMA(13,8)
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values   # SMMA(8,5)
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values    # SMMA(5,3)
    
    # Align Alligator lines to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Load 1d data for Elder Ray (EMA13 of high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    ema13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = ema13 - df_1d['low'].values
    
    # Align Elder Ray to 12h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            continue
        
        # Williams Alligator: Mouth open (JAW > TEETH > LIPS) for uptrend, inverted for downtrend
        alligator_long = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        alligator_short = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power > 0 indicate strength
        bull_power_pos = bull_power_aligned[i] > 0
        bear_power_pos = bear_power_aligned[i] > 0
        
        # Volume spike: current volume > 2x median of last 28 periods
        vol_median = np.median(volume[max(0, i-28):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long entry: Alligator aligned up + Bull Power positive + Volume spike
        if (alligator_long and bull_power_pos and volume_spike and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Alligator aligned down + Bear Power positive + Volume spike
        elif (alligator_short and bear_power_pos and volume_spike and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator closes mouth (TEETH crosses LIPS) or opposite Elder Ray signal
        elif position == 1 and (teeth_aligned[i] <= lips_aligned[i] or bear_power_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (teeth_aligned[i] >= lips_aligned[i] or bull_power_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0