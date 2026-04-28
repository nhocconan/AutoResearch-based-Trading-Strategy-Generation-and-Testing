#!/usr/bin/env python3
"""
12h_WilliamsAlligator_BullishTrend_Filter
Hypothesis: Uses Williams Alligator (13/8/5 SMAs) on daily timeframe to identify bullish/bearish trends. Enters long when price > Alligator Jaw (13-period SMA) and Alligator is in bullish alignment (Teeth > Lips), short when price < Jaw and bearish alignment (Teeth < Lips). Uses volume confirmation (>1.5x 24-period average) to filter false signals. Designed for 12h timeframe to reduce trade frequency and avoid fee drag. Works in bull markets by following trend and in bear markets by shorting downtrends. Targets 15-25 trades/year via strict Alligator alignment + volume filter.
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
    
    # Get daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMAs)
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: >1.5x 24-period average (2 days of 12h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend conditions
        bullish_alignment = teeth_aligned[i] > lips_aligned[i]  # Teeth above Lips
        bearish_alignment = teeth_aligned[i] < lips_aligned[i]  # Teeth below Lips
        
        # Price relative to Jaw (13-period SMA)
        price_above_jaw = close[i] > jaw_aligned[i]
        price_below_jaw = close[i] < jaw_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Entry conditions
        long_entry = price_above_jaw and bullish_alignment and vol_confirm
        short_entry = price_below_jaw and bearish_alignment and vol_confirm
        
        # Exit conditions: price crosses back below/above Teeth (8-period SMA)
        long_exit = close[i] < teeth_aligned[i]
        short_exit = close[i] > teeth_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_BullishTrend_Filter"
timeframe = "12h"
leverage = 1.0