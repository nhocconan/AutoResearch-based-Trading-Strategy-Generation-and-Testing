#!/usr/bin/env python3
"""
4h_1D_Camarilla_R1S1_Breakout_Volume_Tight_V3
Hypothesis: Use 1D Camarilla R1/S1 for directional bias with 4H entry but with tighter filters.
Long when price breaks above daily R1 with volume > 1.5x average and price > SMA200.
Short when price breaks below daily S1 with volume > 1.5x average and price < SMA200.
Reduced position size to 0.20. Added stricter volume and trend filters to reduce trade frequency.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via SMA200 filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Align all daily data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # SMA200 for trend filter
    close_series = pd.Series(close)
    sma200 = close_series.rolling(window=200, min_periods=200).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for SMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(sma200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above SMA200
            if close[i] > r1_aligned[i] and vol_confirm and close[i] > sma200[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume and below SMA200
            elif close[i] < s1_aligned[i] and vol_confirm and close[i] < sma200[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or breaks below SMA200
            if close[i] < r1_aligned[i] or close[i] < sma200[i]:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns above S1 or breaks above SMA200
            if close[i] > s1_aligned[i] or close[i] > sma200[i]:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "4h_1D_Camarilla_R1S1_Breakout_Volume_Tight_V3"
timeframe = "4h"
leverage = 1.0