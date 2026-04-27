#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Williams Alligator (13,8,5 SMAs) from 1-day timeframe with Elder Ray power and volume confirmation.
# Williams Alligator identifies trend direction and strength: Jaw(13), Teeth(8), Lips(5).
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND Volume > 1.5x 20-period average.
# Short when: Lips < Teeth < Jaw (bearish alignment) AND Bear Power > 0 AND Volume > 1.5x 20-period average.
# Exit when Alligator alignment breaks or power fails.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (counter-trend reversals via Elder Ray extremes).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price (High+Low)/2
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Elder Ray: Need EMA13 of close for power calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align Elder Ray powers
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema13_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check Alligator alignment
        bullish_alignment = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        bearish_alignment = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        # Long entry: bullish alignment + positive Bull Power + volume
        if (bullish_alignment and 
            bull_power_1d_aligned[i] > 0 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: bearish alignment + positive Bear Power + volume
        elif (bearish_alignment and 
              bear_power_1d_aligned[i] > 0 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: alignment breaks or power fails
        elif position == 1 and (not bullish_alignment or bull_power_1d_aligned[i] <= 0):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not bearish_alignment or bear_power_1d_aligned[i] <= 0):
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

name = "6h_WilliamsAlligator_ElderRay_VolumeFilter"
timeframe = "6h"
leverage = 1.0