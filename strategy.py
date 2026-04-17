#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R mean reversion + volume confirmation + EMA200 trend filter.
Long when 1d Williams %R < -80 (oversold) with volume > 1.5x 20-period average and price > 6h EMA200.
Short when 1d Williams %R > -20 (overbought) with volume > 1.5x 20-period average and price < 6h EMA200.
Exit when Williams %R crosses back through -50 (mean reversion completion).
Williams %R identifies exhaustion points that work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Volume confirms conviction, EMA200 ensures trend alignment to avoid counter-trend traps.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100.0
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50.0, williams_r)
    
    # Calculate 6h EMA200 for trend filter
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema200[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume and uptrend (price > EMA200)
            if (williams_r_aligned[i] < -80.0 and 
                volume_confirmed and 
                close[i] > ema200[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume and downtrend (price < EMA200)
            elif (williams_r_aligned[i] > -20.0 and 
                  volume_confirmed and 
                  close[i] < ema200[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_MeanReversion_Volume_EMA200_Trend"
timeframe = "6h"
leverage = 1.0