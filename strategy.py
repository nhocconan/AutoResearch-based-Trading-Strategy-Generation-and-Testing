#!/usr/bin/env python3
name = "1d_WilliamsAlligator_Trend_WeeklyFilter"
timeframe = "1d"
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
    
    # Williams Alligator: three SMAs with forward shift
    # Jaw (blue): 13-period SMA, shifted 8 bars forward
    # Teeth (red): 8-period SMA, shifted 5 bars forward  
    # Lips (green): 5-period SMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Weekly SMA for trend filter (8-period)
    sma_8_1w = pd.Series(df_1w['close']).rolling(window=8, min_periods=8).mean().values
    sma_8_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_8_1w)
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(sma_8_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: bullish alignment + price above lips + weekly uptrend + volume
            if bullish_alignment and close[i] > lips[i] and vol_condition and sma_8_1w_aligned[i] > sma_8_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below lips + weekly downtrend + volume
            elif bearish_alignment and close[i] < lips[i] and vol_condition and sma_8_1w_aligned[i] < sma_8_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish alignment or price back below teeth
            if bearish_alignment or close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish alignment or price back above teeth
            if bullish_alignment or close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator on daily timeframe with weekly trend filter
# - Williams Alligator uses three SMAs (5,8,13) with forward shifts to identify trends
# - Bullish: Lips(5) > Teeth(8) > Jaw(13) - aligned upward
# - Bearish: Lips(5) < Teeth(8) < Jaw(13) - aligned downward
# - Weekly SMA8 filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Enter on alignment + price break of lips/middle line with volume
# - Exit when alignment reverses or price crosses middle line
# - Works in both bull and bear markets by following the trend
# - Position size 0.25 targets ~20-60 trades/year to avoid fee drag
# - Alligator is proven trend-following tool that avoids whipsaws in ranging markets
# - Weekly filter ensures we only trade in the direction of the higher timeframe trend
# - Novel combination: Williams Alligator + weekly trend + volume filter not recently tried
# - Aims for 40-120 total trades over 4 years (10-30/year) to stay within limits