#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray force index and volume confirmation.
# Long when price is above Alligator teeth (green line), 1d Elder Ray > 0, and volume > 1.5x 20-period average.
# Short when price is below Alligator teeth, 1d Elder Ray < 0, and volume > 1.5x 20-period average.
# Exit when price crosses back below/above the Alligator lips (red/blue lines).
# Williams Alligator identifies trend presence and direction. Elder Ray confirms bull/bear power.
# Volume filter ensures institutional participation. Designed for low-frequency, high-conviction trades.

name = "12h_WilliamsAlligator_1dElderRay_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray force index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Elder Ray Force Index (13-period EMA of price change * volume)
    price_change = df_1d['close'].diff(1).values
    vol_1d = df_1d['volume'].values
    force_raw = price_change * vol_1d
    elder_ray = pd.Series(force_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    elder_ray_aligned = align_htf_to_ltf(prices, df_1d, elder_ray)
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw (blue): 13-period SMMA shifted 8 bars forward
    # Teeth (red): 8-period SMMA shifted 5 bars forward  
    # Lips (green): 5-period SMMA shifted 3 bars forward
    smma_13 = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    smma_8 = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    smma_5 = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    jaw = np.roll(smma_13, 8)  # shifted 8 bars forward
    teeth = np.roll(smma_8, 5)  # shifted 5 bars forward
    lips = np.roll(smma_5, 3)   # shifted 3 bars forward
    
    # Align to 12h timeframe (no additional delay needed as SMMA uses past data)
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 20) + 8  # max period + jaw shift
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(elder_ray_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above teeth, Elder Ray positive, volume filter
            long_cond = (close[i] > teeth_aligned[i]) and (elder_ray_aligned[i] > 0) and volume_filter[i]
            # Short conditions: price below teeth, Elder Ray negative, volume filter
            short_cond = (close[i] < teeth_aligned[i]) and (elder_ray_aligned[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lips (green line)
            if close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above lips (green line)
            if close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals