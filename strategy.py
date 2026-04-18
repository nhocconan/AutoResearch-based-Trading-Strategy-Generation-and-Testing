#!/usr/bin/env python3
"""
12h_1W_R1S1_Breakout_Volume_V1
Hypothesis: Use weekly (1W) Camarilla R1/S1 for directional bias, 12H for entry with volume confirmation.
Long when price breaks above weekly R1 with volume > 1.5x average and price above 12H EMA50.
Short when price breaks below weekly S1 with volume > 1.5x average and price below 12H EMA50.
Weekly pivot provides stronger trend context for 12H trading, reducing whipsaw.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
Works in bull/bear via weekly trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC for Camarilla calculation
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]  # first week uses same week
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1w = prev_high - prev_low
    r1 = prev_close + range_1w * 1.1 / 12
    s1 = prev_close - range_1w * 1.1 / 12
    
    # Align weekly data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # 12H EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        price_above_ema = close[i] > ema50[i]
        price_below_ema = close[i] < ema50[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume and above EMA50
            if close[i] > r1_aligned[i] and vol_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume and below EMA50
            elif close[i] < s1_aligned[i] and vol_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly R1 or below EMA50
            if close[i] < r1_aligned[i] or close[i] < ema50[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly S1 or above EMA50
            if close[i] > s1_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1W_R1S1_Breakout_Volume_V1"
timeframe = "12h"
leverage = 1.0