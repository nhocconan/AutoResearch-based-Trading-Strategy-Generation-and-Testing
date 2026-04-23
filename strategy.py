#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator trend filter with 1w/1d HTF confluence and volume spike confirmation.
- Long: Price > Alligator Jaw (13) AND Jaw > Teeth (8) AND Teeth > Lips (5) AND volume > 2.0x 20-period avg
- Short: Price < Alligator Jaw AND Jaw < Teeth AND Teeth < Lips AND volume > 2.0x 20-period avg
- Exit: Opposite Alligator alignment OR volume drops below average
- Uses 1w HTF for major trend filter (price > 1w EMA50 for longs, < for shorts)
- Uses 1d HTF for Alligator calculation (smoothed SMAs)
- Designed for very low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
- Alligator provides smooth trend detection with built-in smoothing to reduce whipsaws
- Volume confirmation filters low-conviction moves
- Works in both bull (trend following) and bear (avoids false signals in chop) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for major trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Alligator components from 1d data (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs of median price
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 50 for 1w EMA, 20 for volume MA, 13 for Jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Alligator alignment signals
        # Bullish alignment: Price > Jaw AND Jaw > Teeth AND Teeth > Lips
        bullish_aligned = (close[i] > jaw_aligned[i]) and (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        # Bearish alignment: Price < Jaw AND Jaw < Teeth AND Teeth < Lips
        bearish_aligned = (close[i] < jaw_aligned[i]) and (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        
        if position == 0:
            # Long: Bullish Alligator alignment AND price > 1w EMA50 AND volume confirmation
            if bullish_aligned and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND price < 1w EMA50 AND volume confirmation
            elif bearish_aligned and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR price < 1w EMA50 (major trend break) OR volume drops
            if bearish_aligned or close[i] < ema_50_1w_aligned[i] or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR price > 1w EMA50 (major trend break) OR volume drops
            if bullish_aligned or close[i] > ema_50_1w_aligned[i] or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0