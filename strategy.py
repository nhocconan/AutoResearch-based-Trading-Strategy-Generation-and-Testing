#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-day Bollinger Bands squeeze + 1-week volume surge breakout.
# Long when: price closes above upper Bollinger Band, BB width at 1-year low, volume > 3x 20-period average
# Short when: price closes below lower Bollinger Band, BB width at 1-year low, volume > 3x 20-period average
# Exit when price returns to the 20-period SMA.
# Designed to catch volatility breakouts after low volatility periods in both bull and bear markets.
# Target: 15-25 trades/year per symbol.
name = "12h_BollingerSqueeze_VolumeSurge"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2) on daily data
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # 1-year low of BB width (252 trading days)
    bb_width_min_252 = pd.Series(bb_width_1d).rolling(window=252, min_periods=252).min().values
    bb_squeeze = bb_width_1d <= bb_width_min_252 * 1.1  # within 10% of yearly low
    
    # Align BB squeeze and bands to 12h timeframe
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # 1-week data for volume surge filter
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # 20-week average volume for surge detection
    vol_avg_20w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume_1w > 3 * vol_avg_20w  # 3x surge
    
    # Align volume surge to 12h timeframe
    vol_surge_aligned = align_htf_to_ltf(prices, df_1w, vol_surge)
    
    # 20-period SMA on 12h data for exit
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_surge_aligned[i]) or 
            np.isnan(sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price closes above upper BB, BB squeeze, volume surge
            if price > upper_bb_aligned[i] and bb_squeeze_aligned[i] and vol_surge_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below lower BB, BB squeeze, volume surge
            elif price < lower_bb_aligned[i] and bb_squeeze_aligned[i] and vol_surge_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to 20-period SMA
            if price <= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 20-period SMA
            if price >= sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals