#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA trend filter and volume confirmation
# Williams Alligator consists of three SMAs: Jaw (13), Teeth (8), Lips (5)
# In trending markets: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
# In ranging markets: lines intertwine
# Strategy: Go long when Alligator is bullish aligned AND price > 1d EMA(50)
# Go short when Alligator is bearish aligned AND price < 1d EMA(50)
# Exit when alignment breaks or price crosses 1d EMA
# Volume filter: current volume > 1.5x 20-period average
# Target: 50-150 trades over 4 years by requiring strong alignment + trend + volume

name = "6h_williams_alligator_1dema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h: SMAs of median price
    median_price = (high + low) / 2
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values  # 5-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values  # 8-period
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Wait for Jaw to stabilize
        # Skip if required data not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check Alligator alignment
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 1:  # long position
            # Exit: Alligator loses bullish alignment OR price < 1d EMA(50)
            if not bullish_aligned or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator loses bearish alignment OR price > 1d EMA(50)
            if not bearish_aligned or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator aligned + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if bullish_aligned and close[i] > ema_50_aligned[i]:
                    # Bullish Alligator alignment + above daily EMA
                    signals[i] = 0.25
                    position = 1
                elif bearish_aligned and close[i] < ema_50_aligned[i]:
                    # Bearish Alligator alignment + below daily EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals