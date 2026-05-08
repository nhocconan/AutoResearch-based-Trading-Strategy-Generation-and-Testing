#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMA) < teeth (8-period SMA) < lips (5-period SMA) AND price > lips AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when Alligator jaws > teeth > lips AND price < lips AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back through the teeth (8-period SMA).
# The Alligator identifies trends via SMAs with forward shift, reducing whipsaw. EMA50 filters higher timeframe trend. Volume confirms participation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator components (5, 8, 13 period SMAs with forward shift)
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 2)  # Sufficient warmup for Alligator and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment), price > Lips, EMA50 rising, volume filter
            long_cond = (lips[i] > teeth[i]) and (teeth[i] > jaw[i]) and (close[i] > lips[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: Jaw > Teeth > Lips (bearish alignment), price < Lips, EMA50 falling, volume filter
            short_cond = (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and (close[i] < lips[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Teeth
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Teeth
            if close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals