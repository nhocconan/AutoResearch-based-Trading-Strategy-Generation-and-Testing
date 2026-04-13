#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with daily volume confirmation.
# Uses Williams Alligator (3 SMAs) to identify trends and avoid whipsaws.
# Daily volume ensures breakouts have conviction. Works in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    close_1d = df_1d['close'].values
    
    # Jaw: SMA(13) shifted 8 bars forward
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward
    jaw_values = jaw.values
    
    # Teeth: SMA(8) shifted 5 bars forward
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward
    teeth_values = teeth.values
    
    # Lips: SMA(5) shifted 3 bars forward
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward
    lips_values = lips.values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12-hour timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_values)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x daily volume MA (adjusted for 12h)
        # 2 12h periods per day, so daily MA/2 = approximate 12h period MA
        volume_12h_approx_ma = volume_ma_20_1d_aligned[i] / 2
        volume_condition = volume[i] > (volume_12h_approx_ma * 1.5)
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        # Entry conditions: Alligator alignment with volume confirmation
        # Long when Lips > Teeth > Jaw (bullish alignment) with volume
        # Short when Lips < Teeth < Jaw (bearish alignment) with volume
        if position == 0:
            if lips_above_teeth and teeth_above_jaw and volume_condition:
                position = 1
                signals[i] = position_size
            elif lips_below_teeth and teeth_below_jaw and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if not (lips_above_teeth and teeth_above_jaw):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if not (lips_below_teeth and teeth_below_jaw):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0