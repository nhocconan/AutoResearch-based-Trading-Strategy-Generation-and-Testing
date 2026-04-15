#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Range with Volume Confirmation
# Uses weekly pivot points (calculated from previous week's OHLC) to define support/resistance zones.
# Long when price bounces above weekly pivot with volume confirmation (>1.5x 24-bar median volume).
# Short when price breaks below weekly support with volume confirmation.
# Designed to capture mean reversion at key weekly levels in both bull and bear markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # For each day, we need the prior week's data (Monday to Sunday)
    # We'll use a simplified approach: weekly pivot based on prior 5 trading days
    # In practice, we calculate: P = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # Then R1 = 2*P - Prior Week Low, S1 = 2*P - Prior Week High
    # R2 = P + (Prior Week High - Prior Week Low), S2 = P - (Prior Week High - Prior Week Low)
    
    # Calculate rolling weekly high/low/close (5-day lookback for prior week)
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1)  # prior week
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1)    # prior week
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1)  # prior week
    
    # Weekly pivot and support/resistance levels
    pivot = (week_high + week_low + week_close) / 3
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    
    # Align weekly levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # Volume confirmation: current > 1.5x median of last 24 bars (4 days of 6h data)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r1_6h[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Long: Price above weekly pivot (S1) with volume confirmation, and not too far above R1
        # Enter long when price crosses above S1 with volume, targeting pivot/R1
        if (price > s1_6h[i] and 
            price <= r1_6h[i] and  # Not too far extended
            vol > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price below weekly pivot (R1) with volume confirmation, and not too far below S1
        # Enter short when price crosses below R1 with volume, targeting pivot/S1
        elif (price < r1_6h[i] and 
              price >= s1_6h[i] and  # Not too far extended
              vol > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Price returns to pivot zone or volume dries up
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (price <= pivot_6h[i] or vol <= vol_threshold[i])) or
               (signals[i-1] == -0.25 and (price >= pivot_6h[i] or vol <= vol_threshold[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyPivot_Range_Volume"
timeframe = "6h"
leverage = 1.0