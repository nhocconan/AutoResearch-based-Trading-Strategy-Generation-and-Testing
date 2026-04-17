#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Mean Reversion with 1w EMA Trend Filter and Volume Spike.
Long when Williams %R(14) < -80 (oversold) AND price > 1w EMA34 (bullish trend) AND 6h volume > 2.0x 20-bar average.
Short when Williams %R(14) > -20 (overbought) AND price < 1w EMA34 (bearish trend) AND 6h volume > 2.0x 20-bar average.
Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
Uses 1w for trend filter and 6h for execution and volume confirmation.
Designed to capture mean-reversion bounces within the dominant weekly trend with volume confirmation.
Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) > 0, williams_r, -50.0)
    
    # Calculate 6h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        if position == 0:
            # Long: oversold + bullish trend + volume confirmation
            if (oversold and close[i] > ema34_1w_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: overbought + bearish trend + volume confirmation
            elif (overbought and close[i] < ema34_1w_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1wEMA34_Volume"
timeframe = "6h"
leverage = 1.0