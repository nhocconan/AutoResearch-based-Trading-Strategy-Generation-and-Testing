#!/usr/bin/env python3
name = "1d_Williams_Alligator_ElderRay_Trend_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close):
    """Williams Alligator: 3 SMAs (Jaw 13, Teeth 8, Lips 5) with forward shift"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Median price
    median_price = (high_s + low_s) / 2
    
    # Alligator lines
    jaw = median_price.rolling(window=13, min_periods=13).mean().shift(8)   # Blue line
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5)    # Red line
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3)     # Green line
    
    return jaw.values, teeth.values, lips.values

def elder_ray(high, low, close):
    """Elder Ray: Bull Power (High - EMA13), Bear Power (Low - EMA13)"""
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean()
    
    bull_power = high - ema13.values
    bear_power = low - ema13.values
    
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Alligator on weekly
    jaw_1w, teeth_1w, lips_1w = williams_alligator(high_1w, low_1w, close_1w)
    
    # Elder Ray on weekly
    bull_power_1w, bear_power_1w = elder_ray(high_1w, low_1w, close_1w)
    
    # Align weekly indicators to daily
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    bull_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bull_power_1w)
    bear_power_1w_aligned = align_htf_to_ltf(prices, df_1w, bear_power_1w)
    
    # Daily volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or 
            np.isnan(lips_1w_aligned[i]) or np.isnan(bull_power_1w_aligned[i]) or
            np.isnan(bear_power_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw), Bull Power positive, volume spike
            if (lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i] and
                bull_power_1w_aligned[i] > 0 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator inverted (Lips < Teeth < Jaw), Bear Power negative, volume spike
            elif (lips_1w_aligned[i] < teeth_1w_aligned[i] < jaw_1w_aligned[i] and
                  bear_power_1w_aligned[i] < 0 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator misalignment or Bear Power negative
            if (lips_1w_aligned[i] <= teeth_1w_aligned[i] or 
                bear_power_1w_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator misalignment or Bull Power positive
            if (lips_1w_aligned[i] >= teeth_1w_aligned[i] or 
                bull_power_1w_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals