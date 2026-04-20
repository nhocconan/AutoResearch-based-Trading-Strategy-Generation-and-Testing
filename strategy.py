#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w Volume Trend Filter
# - Williams Alligator (13,8,5 SMAs) on 1d for trend direction and entry signals
# - Long when Jaw < Teeth < Lips (bullish alignment) and price > Lips
# - Short when Jaw > Teeth > Lips (bearish alignment) and price < Lips
# - Volume filter: 1w average volume > 1.5x 4w average volume to confirm institutional interest
# - Designed for 1d timeframe with low frequency to avoid overtrading
# - Target: 10-25 trades per year per symbol (40-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2  # Using median price as per Alligator definition
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_raw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_raw = pd.Series(median_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_raw = pd.Series(median_1d).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Load 1w data for volume trend filter
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate volume trend: current 1w average volume vs 4w average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=1, min_periods=1).mean()  # current week
    vol_ma_4w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean()  # 4-week average
    volume_ratio = vol_ma_1w / vol_ma_4w
    
    # Align indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if NaN in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ratio = volume_ratio_aligned[i]
        
        # Volume filter: require above-average volume for confirmation
        volume_ok = vol_ratio > 1.2
        
        if position == 0:
            # Long entry: Bullish alignment (Jaw < Teeth < Lips) and price > Lips
            if (jaw_val < teeth_val < lips_val) and (price > lips_val) and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment (Jaw > Teeth > Lips) and price < Lips
            elif (jaw_val > teeth_val > lips_val) and (price < lips_val) and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or price crosses below Teeth
            if (jaw_val > teeth_val > lips_val) or (price < teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or price crosses above Teeth
            if (jaw_val < teeth_val < lips_val) or (price > teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wVolumeTrend"
timeframe = "1d"
leverage = 1.0