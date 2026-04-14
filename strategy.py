#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with Elder Ray power and volume confirmation
# Long when green line > red line (bullish alignment) AND bull power > 0 AND volume > 1.5x average
# Short when red line > green line (bearish alignment) AND bear power > 0 AND volume > 1.5x average
# Exit when alignment reverses or volume drops below average
# Uses Alligator for trend direction, Elder Ray for power confirmation, volume for conviction
# Target: 60-120 total trades over 4 years (15-30/year)

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator lines (13,8,5 smoothed with future shift)
    # Jaw (13-period SMMA shifted 8 bars)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (8-period SMMA shifted 5 bars)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (5-period SMMA shifted 3 bars)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max shift is 8 for jaw)
    start = 20  # for 20-period volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: lips > teeth > jaw = bullish, jaw > teeth > lips = bearish
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        vol_condition = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Long setup: bullish alignment AND bull power > 0 AND volume spike
            if bullish_alignment and (bull_power_aligned[i] > 0) and vol_condition:
                position = 1
                signals[i] = position_size
            # Short setup: bearish alignment AND bear power > 0 AND volume spike
            elif bearish_alignment and (bear_power_aligned[i] > 0) and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment OR bull power <= 0 OR volume drops
            if (not bullish_alignment) or (bull_power_aligned[i] <= 0) or (not vol_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish alignment OR bear power <= 0 OR volume drops
            if (not bearish_alignment) or (bear_power_aligned[i] <= 0) or (not vol_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Alligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0