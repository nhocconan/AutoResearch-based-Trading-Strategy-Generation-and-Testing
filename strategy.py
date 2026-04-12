#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_regime_v1
Hypothesis: 12-hour trading using Camarilla levels from daily timeframe with volume confirmation and Choppiness Index regime filter.
Works in bull/bear markets by combining mean-reversion at extreme levels (S4/R4) with trend-following in strong markets.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

name = "12h_1d_camarilla_breakout_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla, ATR and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (for Camarilla)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # ATR for volatility filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    
    # Align all indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Market regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if is_ranging:
            # Ranging market: mean reversion at extreme levels
            # Long: price touches S4 support
            if (close[i] <= s4_aligned[i] and vol_confirm[i] and 
                atr_aligned[i] > 0 and position != 1):
                position = 1
                signals[i] = 0.25
            # Short: price touches R4 resistance
            elif (close[i] >= r4_aligned[i] and vol_confirm[i] and 
                  atr_aligned[i] > 0 and position != -1):
                position = -1
                signals[i] = -0.25
            # Exit: price returns to S3/R3 levels
            elif position == 1 and close[i] >= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] <= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
        else:
            # Trending market: breakout continuation
            # Long: price breaks above R4 with volume
            if (close[i] > r4_aligned[i] and vol_confirm[i] and 
                atr_aligned[i] > 0 and position != 1):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume
            elif (close[i] < s4_aligned[i] and vol_confirm[i] and 
                  atr_aligned[i] > 0 and position != -1):
                position = -1
                signals[i] = -0.25
            # Exit: reverse signal
            elif position == 1 and close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
        
        # Hold current position
        if position == 1 and signals[i] == 0.0:
            signals[i] = 0.25
        elif position == -1 and signals[i] == 0.0:
            signals[i] = -0.25
    
    return signals