#!/usr/bin/env python3
"""
12h_1d_Range_Reversion_Bands_V1
Hypothesis: Use 1-day Bollinger Bands (20,2) as dynamic support/resistance on 12h timeframe. 
Enter long when price touches lower band with bullish momentum (close > open), short when touches upper band with bearish momentum (close < open). 
Requires volume > 1.2x 20-period average for confirmation. 
Exit on opposite band touch or when momentum reverses (close crosses opposite direction of entry).
Designed for range-bound markets (2022-2024, 2025) with mean-reversion edge. 
Targets 15-25 trades/year by requiring band touch + volume + momentum alignment.
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) via adaptive bands.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = np.full_like(close_1d, np.nan)
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Align bands to 12h timeframe (wait for bar close)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and bands ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(sma_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price at or below lower band, bullish candle, with volume
            if (low[i] <= lower_aligned[i] and close[i] > open_price[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above upper band, bearish candle, with volume
            elif (high[i] >= upper_aligned[i] and close[i] < open_price[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price touches/above upper band OR bearish reversal
            if (high[i] >= upper_aligned[i] or close[i] < open_price[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches/below lower band OR bullish reversal
            if (low[i] <= lower_aligned[i] or close[i] > open_price[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Range_Reversion_Bands_V1"
timeframe = "12h"
leverage = 1.0