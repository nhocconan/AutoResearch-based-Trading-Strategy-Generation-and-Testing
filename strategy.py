#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and absence of trend (choppy market).
# In chop (Alligator lines intertwined), we fade extremes using volume spikes at Bollinger Bands.
# In trend (Alligator aligned), we breakout in direction of trend with volume confirmation.
# Uses 12h primary timeframe as specified in experiment #124328 with 1w/1d HTF.
# Designed for 12-37 trades/year to minimize fee drag and work in both bull and bear markets.

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator, Bollinger Bands, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Bollinger Bands on 1d close (20, 2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Align Bollinger Bands
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Choppiness Index (CHOP) on 1d - measures trend vs range
    # CHOP > 61.8 = ranging/chop, CHOP < 38.2 = trending
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    sum_tr = atr_1d * 14
    range_14 = highest_high_1d - lowest_low_1d
    chop = 100 * np.log10(sum_tr / range_14) / np.log10(14)
    chop[range_14 == 0] = 100  # Handle zero range
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 20-period EMA of volume on 1d
    vol_ema_20_1d = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_ema_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime determination based on Chop and Alligator alignment
        is_chop = chop_aligned[i] > 61.8
        is_trend = chop_aligned[i] < 38.2
        alligator_aligned = (
            (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or  # Uptrend aligned
            (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])     # Downtrend aligned
        )
        
        if position == 0:
            # Enter long in chop: price at lower BB with volume spike
            if is_chop and close[i] <= bb_lower_aligned[i] and volume[i] > (2.0 * vol_ema_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short in chop: price at upper BB with volume spike
            elif is_chop and close[i] >= bb_upper_aligned[i] and volume[i] > (2.0 * vol_ema_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            # Enter long in trend: Alligator aligned up AND price > middle BB with volume spike
            elif is_trend and alligator_aligned and lips_aligned[i] > teeth_aligned[i] and close[i] > bb_middle_aligned[i] and volume[i] > (2.0 * vol_ema_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short in trend: Alligator aligned down AND price < middle BB with volume spike
            elif is_trend and alligator_aligned and lips_aligned[i] < teeth_aligned[i] and close[i] < bb_middle_aligned[i] and volume[i] > (2.0 * vol_ema_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses middle BB OR Alligator reverses
            if close[i] < bb_middle_aligned[i] or not (lips_aligned[i] > teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses middle BB OR Alligator reverses
            if close[i] > bb_middle_aligned[i] or not (lips_aligned[i] < teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals