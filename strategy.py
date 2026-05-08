#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Williams_Alligator_ElderRay_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator components (13, 8, 5 SMAs with future shifts)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Shift jaws/teeth/lips by future bars (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)   # shift forward 8 bars
    teeth_shifted = np.roll(teeth, 5) # shift forward 5 bars
    lips_shifted = np.roll(lips, 3)   # shift forward 3 bars
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (df_1d['high'].values - ema13)
    bear_power = (df_1d['low'].values - ema13)
    
    # Align Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator: Mouth open = jaws > teeth > lips (bullish) or jaws < teeth < lips (bearish)
        # Elder Ray: Bull power > 0 and rising, Bear power < 0 and falling
        
        if position == 0:
            # Long: Alligator bullish alignment + bull power positive and rising + volume
            if (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                bull_power_aligned[i] > 0 and
                i > start_idx and bull_power_aligned[i] > bull_power_aligned[i-1] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment + bear power negative and falling + volume
            elif (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                  bear_power_aligned[i] < 0 and
                  i > start_idx and bear_power_aligned[i] < bear_power_aligned[i-1] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator changes direction OR bull power turns negative
            if (jaw_aligned[i] <= teeth_aligned[i] or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator changes direction OR bear power turns positive
            if (jaw_aligned[i] >= teeth_aligned[i] or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals