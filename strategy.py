#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) with volume confirmation
# Uses Williams Alligator (13,8,5 SMAs) on 6h to detect trend direction
# Uses Elder Ray from 1d: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# In bullish alignment (Green > Red > Blue Alligator + Bull Power > 0): long bias
# In bearish alignment (Blue > Red > Green Alligator + Bear Power > 0): short bias
# Volume filter: require volume > 1.3x 20-period EMA to avoid false signals
# Designed for ~15-30 trades/year with strong trend filtering for both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 6h (13,8,5 SMAs)
    # Jaw (13-period), Teeth (8-period), Lips (5-period)
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean()  # Blue line
    teeth = close_series.rolling(window=8, min_periods=8).mean()   # Red line
    lips = close_series.rolling(window=5, min_periods=5).mean()    # Green line
    
    # Calculate Elder Ray on 1d (Bull Power, Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 6s timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)[0]  # We'll get the aligned arrays properly below
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)[0]
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)[0]
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power, additional_delay_bars=0)[0]  # Elder Ray uses same bar
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power, additional_delay_bars=0)[0]
    
    # Properly get all aligned arrays
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any values are NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Williams Alligator alignment check
        # Bullish: Lips > Teeth > Jaw (Green > Red > Blue)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish: Jaw > Teeth > Lips (Blue > Red > Green)
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Elder Ray confirmation
        bullish_power = bull_power_aligned[i] > 0
        bearish_power = bear_power_aligned[i] > 0
        
        # Entry conditions
        if position == 0 and bullish_alignment and bullish_power and volume_confirm:
            position = 1
            signals[i] = position_size
        elif position == 0 and bearish_alignment and bearish_power and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit conditions: when alignment breaks or power fades
        elif position == 1 and (not bullish_alignment or not bullish_power):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or not bearish_power):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Alligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0