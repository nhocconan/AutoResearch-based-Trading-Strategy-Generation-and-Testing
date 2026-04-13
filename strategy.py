#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with price crossover and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw, Teeth, Lips) that act as dynamic support/resistance.
# In trending markets, price stays aligned with the Alligator; in ranging markets, the lines intertwine.
# We use the crossover of Lips (fastest) and Teeth (middle) as entry signal, confirmed by Jaw direction.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets by
# following the trend direction defined by the Alligator alignment.
# Target: 15-30 trades per year (60-120 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Williams Alligator (13,8,5 SMAs on median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate median price for Alligator
    median_price = (df_1d['high'] + df_1d['low']) / 2
    
    # Williams Alligator lines: Jaw (13-period), Teeth (8-period), Lips (5-period) SMAs
    jaw = np.full(len(df_1d), np.nan)
    teeth = np.full(len(df_1d), np.nan)
    lips = np.full(len(df_1d), np.nan)
    
    # Calculate SMAs with proper handling of NaN
    for i in range(len(df_1d)):
        if i >= 12:  # Jaw: 13-period SMA
            jaw[i] = np.nanmean(median_price[i-12:i+1])
        if i >= 7:   # Teeth: 8-period SMA
            teeth[i] = np.nanmean(median_price[i-7:i+1])
        if i >= 4:   # Lips: 5-period SMA
            lips[i] = np.nanmean(median_price[i-4:i+1])
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Lips crosses above Teeth AND Jaw < Teeth (bullish alignment)
            if (lips_aligned[i] > teeth_aligned[i] and 
                lips_aligned[i-1] <= teeth_aligned[i-1] and  # crossover just happened
                jaw_aligned[i] < teeth_aligned[i] and        # bullish alignment: Jaw below Teeth
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Lips crosses below Teeth AND Jaw > Teeth (bearish alignment)
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  lips_aligned[i-1] >= teeth_aligned[i-1] and  # crossover just happened
                  jaw_aligned[i] > teeth_aligned[i] and        # bearish alignment: Jaw above Teeth
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips crosses back below Teeth OR Jaw crosses above Teeth (trend change)
            if (lips_aligned[i] < teeth_aligned[i] and lips_aligned[i-1] >= teeth_aligned[i-1]) or \
               (jaw_aligned[i] > teeth_aligned[i] and jaw_aligned[i-1] <= teeth_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Lips crosses back above Teeth OR Jaw crosses below Teeth (trend change)
            if (lips_aligned[i] > teeth_aligned[i] and lips_aligned[i-1] <= teeth_aligned[i-1]) or \
               (jaw_aligned[i] < teeth_aligned[i] and jaw_aligned[i-1] >= teeth_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0