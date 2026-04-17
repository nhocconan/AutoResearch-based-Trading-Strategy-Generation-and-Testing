#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R mean reversion + volume spike + 4h EMA50 trend filter.
Long when Williams %R < -80 (oversold) + volume > 2x 20-period average + price > EMA50 (uptrend context).
Short when Williams %R > -20 (overbought) + volume > 2x 20-period average + price < EMA50 (downtrend context).
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies extreme price levels; volume confirms institutional participation; EMA50 filters for trend alignment.
Designed to work in both bull and bear markets by fading extremes with volume confirmation and trend context.
Uses 1d timeframe for Williams %R (reduces noise) and 4h for entry timing and trend filter.
Target: 20-40 trades/year per symbol to minimize fee drag while capturing high-probability reversals.
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
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 4h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volume confirmation + uptrend (price > EMA50)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume confirmation + downtrend (price < EMA50)
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (recovering from oversold)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (recovering from overbought)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_MeanReversion_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0