#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume spike confirmation.
# Long when price > Alligator Jaw AND Jaw > Teeth AND Teeth > Lips AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when price < Alligator Jaw AND Jaw < Teeth AND Teeth < Lips AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Alligator mouth (between Jaw and Lips).
# Williams Alligator identifies trends via smoothed SMAs (Jaw=13, Teeth=8, Lips=5).
# Trend filter ensures alignment with higher timeframe direction.
# Volume confirmation avoids false signals. Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # Williams Alligator components (13, 8, 5 period SMAs)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
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
    
    start_idx = max(50, 13)  # Sufficient warmup for EMA50 and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND Jaw > Teeth AND Teeth > Lips AND 1d EMA50 rising AND volume filter
            long_cond = (close[i] > jaw[i]) and (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price < Jaw AND Jaw < Teeth AND Teeth < Lips AND 1d EMA50 falling AND volume filter
            short_cond = (close[i] < jaw[i]) and (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Lips (Alligator mouth)
            if close[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Lips (Alligator mouth)
            if close[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals