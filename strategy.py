#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
# Long: Price > Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 + volume > 1.5x 20-period EMA
# Short: Price < Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 + volume > 1.5x 20-period EMA
# Uses 6h timeframe to reduce noise, 1d for trend filter, volume confirmation to avoid false signals
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1d trend direction and using Alligator for structure

name = "6h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h timeframe (Jaw=13, Teeth=8, Lips=5)
    # All lines are smoothed with future values, so we need to shift by 5 to avoid look-ahead
    # Median prices: (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 20-period EMA on 6h volume
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
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams Alligator signals
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish alignment + price > 1d EMA50 + volume spike
            if bullish_alignment and close[i] > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < 1d EMA50 + volume spike
            elif bearish_alignment and close[i] < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price < 1d EMA50 (trend change)
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR price > 1d EMA50 (trend change)
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals