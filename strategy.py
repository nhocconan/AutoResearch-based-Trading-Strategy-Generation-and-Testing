#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trendless periods when lines are intertwined
# Trades only when Alligator is "awake" (lines separated) in direction of 1d EMA50
# Volume confirmation requires 1.5x average volume to ensure participation
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# Works in bull markets by catching trends, avoids whipsaws in ranging markets via Alligator sleep filter

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
    
    # Williams Alligator on 6h: SMAs of median price
    # Jaw: 13-period SMA, 8 bars ahead
    # Teeth: 8-period SMA, 5 bars ahead  
    # Lips: 5-period SMA, 3 bars ahead
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
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
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Alligator sleeping: lines intertwined (no clear trend)
        # Sleeping condition: max difference between lines < 0.1% of price
        max_diff = max(abs(jaw[i] - teeth[i]), abs(teeth[i] - lips[i]), abs(lips[i] - jaw[i]))
        sleeping = max_diff < (0.001 * close[i])
        
        # Alligator awake and trending up: Lips > Teeth > Jaw
        trending_up = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Alligator awake and trending down: Jaw > Teeth > Lips
        trending_down = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Entry signals
        if position == 0:
            # Long: Alligator awake + trending up + price above Jaw + volume spike + price above 1d EMA50
            if (not sleeping and trending_up and close[i] > jaw[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator awake + trending down + price below Jaw + volume spike + price below 1d EMA50
            elif (not sleeping and trending_down and close[i] < jaw[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator starts sleeping OR trend reverses OR price below 1d EMA50
            if sleeping or not trending_up or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator starts sleeping OR trend reverses OR price above 1d EMA50
            if sleeping or not trending_down or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals