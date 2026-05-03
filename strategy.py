#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator identifies trend direction via smoothed medians (Jaw/Teeth/Lips)
# 1d EMA50 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation requires 1.5x average volume to ensure participation while avoiding overtrading
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by following the 1d trend direction and using Alligator for entry timing

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components on 12h timeframe
    # Jaw: 13-period SMMA, offset 8 bars
    # Teeth: 8-period SMMA, offset 5 bars  
    # Lips: 5-period SMMA, offset 3 bars
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    close_series = pd.Series(close)
    jaw = close_series.ewm(alpha=1/13, adjust=False).mean().shift(8).values
    teeth = close_series.ewm(alpha=1/8, adjust=False).mean().shift(5).values
    lips = close_series.ewm(alpha=1/5, adjust=False).mean().shift(3).values
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Alligator is bullish when Lips > Teeth > Jaw (green alignment)
        # Alligator is bearish when Jaw > Teeth > Lips (red alignment)
        # Long: Alligator bullish + volume spike + price above 1d EMA50
        # Short: Alligator bearish + volume spike + price below 1d EMA50
        if position == 0:
            if (lips[i] > teeth[i] > jaw[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (jaw[i] > teeth[i] > lips[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR price below 1d EMA50
            if not (lips[i] > teeth[i] > jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR price above 1d EMA50
            if not (jaw[i] > teeth[i] > lips[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals