#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator trend filter with 1w Bollinger Band squeeze and volume spike
# Long when price > Alligator's Jaw, BB width < 20th percentile, and volume > 1.5x average
# Short when price < Alligator's Jaw, BB width < 20th percentile, and volume > 1.5x average
# Exit when price crosses Alligator's Teeth or BB width > 50th percentile
# Uses Alligator for trend, Bollinger Bands for volatility regime, volume for confirmation
# Designed to catch strong trends during low volatility periods in both bull and bear markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_Williams_Alligator_BB_Squeeze_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 1w Williams Alligator (Jaw, Teeth, Lips)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Alligator components: Smoothed Moving Average (SMA with smoothing)
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Jaw is the main trend indicator (blue line)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    
    # Teeth and Lips for exit signals
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1w Bollinger Band width for volatility regime
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean()
    std_dev = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std()
    upper = ma + (std_dev * bb_std)
    lower = ma - (std_dev * bb_std)
    bb_width = ((upper - lower) / ma) * 100  # Percentage width
    
    # Align BB width to 12h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width.values)
    
    # Calculate percentiles of BB width for regime filtering
    # We'll calculate rolling percentiles to avoid look-ahead
    bb_width_series = pd.Series(bb_width_aligned)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=10).quantile(0.20).values
    bb_width_50th = bb_width_series.rolling(window=50, min_periods=10).quantile(0.50).values
    
    # Volume confirmation: current volume > 1.5x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(bb_width_20th[i]) or np.isnan(bb_width_50th[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Jaw, BB squeeze (width < 20th percentile), volume spike
            if (close[i] > jaw_aligned[i] and 
                bb_width_aligned[i] < bb_width_20th[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw, BB squeeze (width < 20th percentile), volume spike
            elif (close[i] < jaw_aligned[i] and 
                  bb_width_aligned[i] < bb_width_20th[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Teeth OR BB width expands (> 50th percentile)
            if (close[i] < teeth_aligned[i]) or (bb_width_aligned[i] > bb_width_50th[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Teeth OR BB width expands (> 50th percentile)
            if (close[i] > teeth_aligned[i]) or (bb_width_aligned[i] > bb_width_50th[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals