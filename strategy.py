#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence via smoothed SMAs
# 1d EMA50 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1d trend and using Alligator sleep/awake cycle

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "6h"
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
    
    # Williams Alligator: three smoothed SMAs (Jaw=13, Teeth=8, Lips=5)
    # All values shifted forward by respective amounts: Jaw=8, Teeth=5, Lips=3
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) = Wilder's smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + arr[i]) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components on median price (HL/2)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Shift forward: Jaw by 8, Teeth by 5, Lips by 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become NaN due to roll
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (strict to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Alligator signals:
        # Sleeping: Jaw > Teeth > Lips or Jaw < Teeth < Lips (no trend)
        # Awakening: Lips crosses Teeth or Jaw (trend forming)
        # Trending: Lips > Teeth > Jaw (up) or Lips < Teeth < Jaw (down)
        lips_above_teeth = lips[i] > teeth[i]
        lips_above_jaw = lips[i] > jaw[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        
        lips_below_teeth = lips[i] < teeth[i]
        lips_below_jaw = lips[i] < jaw[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Trending up: Lips > Teeth > Jaw
        trending_up = lips_above_teeth and teeth_above_jaw
        # Trending down: Lips < Teeth < Jaw
        trending_down = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Enter long: Alligator trending up + volume spike + price above 1d EMA50
            if trending_up and volume_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator trending down + volume spike + price below 1d EMA50
            elif trending_down and volume_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator starts sleeping (Lips crosses below Teeth) OR price below 1d EMA50
            if lips[i] < teeth[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator starts sleeping (Lips crosses above Teeth) OR price above 1d EMA50
            if lips[i] > teeth[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals