#!/usr/bin/env python3
# 12h_alligator_trend_volume_v1
# Hypothesis: Uses 12-hour Williams Alligator (SMAs with smoothing) to identify trends, with volume confirmation for entry.
# Long when price > Alligator Jaw (13-period SMA shifted 8 bars) and Teeth > Lips (bullish alignment), short when opposite.
# Includes volume filter (>1.5x 20-period average) to avoid false breakouts. Designed for low trade frequency (~15-30/year)
# to minimize fee drag while capturing major trends in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_alligator_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Alligator (Williams Alligator uses SMAs)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13 SMA, 8 shift), Teeth (8 SMA, 5 shift), Lips (5 SMA, 3 shift)
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 12h timeframe
    jaw = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips = align_htf_to_ltf(prices, df_1d, lips_raw)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend conditions
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        price_above_jaw = close[i] > jaw[i]
        price_below_jaw = close[i] < jaw[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: bearish alignment or price below jaw
            if bearish_alignment or price_below_jaw:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish alignment or price above jaw
            if bullish_alignment or price_above_jaw:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: bullish alignment and price above jaw
                if bullish_alignment and price_above_jaw:
                    position = 1
                    signals[i] = 0.25
                # Short entry: bearish alignment and price below jaw
                elif bearish_alignment and price_below_jaw:
                    position = -1
                    signals[i] = -0.25
    
    return signals