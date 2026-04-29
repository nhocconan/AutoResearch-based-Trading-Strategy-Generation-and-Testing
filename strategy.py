#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when Alligator lines cross (jaws = teeth or jaws = lips) or price retests center line (teeth)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Alligator identifies trend direction and strength; EMA50 filters counter-trend moves on 1d;
# volume confirmation ensures breakout strength. Works in bull via trend continuation,
# in bear via trend continuation. Novelty: using 12h timeframe (lower frequency) with Williams Alligator
# as primary trend filter, reducing false signals vs basic MA crossovers.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 1d data
    # Alligator: Jaw (blue, 13-period SMMA), Teeth (red, 8-period SMMA), Lips (green, 5-period SMMA)
    # SMMA (Smoothed Moving Average): similar to EMA but with different smoothing
    # SMMA formula: SMMA(i) = (SMMA(i-1) * (period-1) + close(i)) / period
    
    close_1d = df_1d['close'].values
    
    # Calculate SMMA for Alligator lines
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    alligator_jaw = smma(close_1d, 13)   # 13-period SMMA
    alligator_teeth = smma(close_1d, 8)   # 8-period SMMA
    alligator_lips = smma(close_1d, 5)    # 5-period SMMA
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 12h timeframe
    alligator_jaw_aligned = align_htf_to_ltf(prices, df_1d, alligator_jaw)
    alligator_teeth_aligned = align_htf_to_ltf(prices, df_1d, alligator_teeth)
    alligator_lips_aligned = align_htf_to_ltf(prices, df_1d, alligator_lips)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(alligator_jaw_aligned[i]) or np.isnan(alligator_teeth_aligned[i]) or 
            np.isnan(alligator_lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_jaw = alligator_jaw_aligned[i]
        curr_teeth = alligator_teeth_aligned[i]
        curr_lips = alligator_lips_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator lines cross (jaws = teeth or jaws = lips) or price retests center line (teeth)
            if (curr_jaw >= curr_teeth or curr_jaw >= curr_lips) or np.abs(curr_close - curr_teeth) < 0.001 * curr_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (jaws = teeth or jaws = lips) or price retests center line (teeth)
            if (curr_jaw <= curr_teeth or curr_jaw <= curr_lips) or np.abs(curr_close - curr_teeth) < 0.001 * curr_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Alligator is bullish (jaws < teeth < lips) AND price > 1d EMA50 AND volume confirmation
            if curr_jaw < curr_teeth and curr_teeth < curr_lips and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator is bearish (jaws > teeth > lips) AND price < 1d EMA50 AND volume confirmation
            elif curr_jaw > curr_teeth and curr_teeth > curr_lips and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals