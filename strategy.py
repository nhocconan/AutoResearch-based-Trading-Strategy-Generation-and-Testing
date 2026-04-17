#!/usr/bin/env python3
"""
4h_EMA21_With_1d_Trend_Filter
Hypothesis: EMA21 provides responsive trend detection; combining with 1d EMA200 reduces whipsaw in choppy markets while capturing strong trends.
Long when EMA21 crosses above EMA50 + price > EMA21 + 1d EMA200 upward slope + volume > 1.5x average.
Short when EMA21 crosses below EMA50 + price < EMA21 + 1d EMA200 downward slope + volume > 1.5x average.
Exit on opposite EMA cross. Position size: ±0.25. Uses 4h primary with 1d trend filter.
Designed to work in bull (trend capture) and bear (avoids false signals via 1d filter).
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
    
    # Calculate EMAs
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA crossovers
    ema21_cross_above_ema50 = (ema21 > ema50) & (np.roll(ema21, 1) <= np.roll(ema50, 1))
    ema21_cross_below_ema50 = (ema21 < ema50) & (np.roll(ema21, 1) >= np.roll(ema50, 1))
    # Handle first element
    ema21_cross_above_ema50[0] = False
    ema21_cross_below_ema50[0] = False
    
    # Volume confirmation (10-period MA on 4h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d EMA200 slope (1-period change)
    ema200_1d_slope = np.diff(ema200_1d, prepend=0)
    ema200_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d_slope)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(21, 50, 10, 200)  # EMA21, EMA50, volume MA10, EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(ema21_cross_above_ema50[i]) or 
            np.isnan(ema21_cross_below_ema50[i]) or 
            np.isnan(volume_ma10[i]) or 
            np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema200_1d_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: EMA21 crosses above EMA50 + price > EMA21 + 1d uptrend + volume filter
            if ema21_cross_above_ema50[i] and close[i] > ema21[i] and ema200_1d_slope_aligned[i] > 0 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: EMA21 crosses below EMA50 + price < EMA21 + 1d downtrend + volume filter
            elif ema21_cross_below_ema50[i] and close[i] < ema21[i] and ema200_1d_slope_aligned[i] < 0 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: EMA21 crosses below EMA50
            if ema21_cross_below_ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA21 crosses above EMA50
            if ema21_cross_above_ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA21_With_1d_Trend_Filter"
timeframe = "4h"
leverage = 1.0