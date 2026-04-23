#!/usr/bin/env python3
"""
Hypothesis: 12-hour price action relative to 1-day Bollinger Bands with volume confirmation.
Long when price touches lower Bollinger Band (20, 2) and volume > 1.5x 20-period average.
Short when price touches upper Bollinger Band (20, 2) and volume > 1.5x 20-period average.
Exit when price crosses the 20-period SMA.
Designed for low-frequency mean reversion in ranging markets with volatility expansion filter.
Works in both bull and bear markets by fading extremes during volatility spikes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align HTF indicators to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_threshold = 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB and volume spike
            if close[i] <= lower_bb_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB and volume spike
            elif close[i] >= upper_bb_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses 20-period SMA
            exit_signal = False
            if position == 1:
                # Exit long: price crosses below SMA
                if close[i] < sma_20_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above SMA
                if close[i] > sma_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Bollinger_Band_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0