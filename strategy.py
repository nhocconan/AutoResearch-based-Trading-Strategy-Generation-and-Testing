#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Volume_Trend
Strategy: Daily breakout of Camarilla R1/S1 levels with weekly trend filter and volume confirmation.
Long: Price breaks above daily R1 + volume > 2x 20-day avg + weekly close > weekly open (bullish week)
Short: Price breaks below daily S1 + volume > 2x 20-day avg + weekly close < weekly open (bearish week)
Exit: Price returns to previous day's close
Position size: 0.25
Designed to capture institutional levels with weekly trend alignment in both bull and bear markets.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    range_val = high - low
    R1 = close + (range_val * 1.1 / 12)
    S1 = close - (range_val * 1.1 / 12)
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla levels
    R1, S1 = calculate_camarilla(high, low, close)
    
    # Calculate 20-day volume average for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Align weekly data to daily timeframe (weekly trend only updates on weekly close)
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, weekly_open)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need volume MA20 and at least 1 day
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(weekly_open_aligned[i]) or 
            np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-day average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Weekly trend filter: bullish/bearish week
        weekly_bullish = weekly_close_aligned[i] > weekly_open_aligned[i]
        weekly_bearish = weekly_close_aligned[i] < weekly_open_aligned[i]
        
        # Breakout conditions (use previous day's levels to avoid look-ahead)
        breakout_up = close[i] > R1[i-1]
        breakout_down = close[i] < S1[i-1]
        
        # Exit condition: return to previous day's close
        return_to_prev_close = abs(close[i] - close[i-1]) < (0.005 * close[i])  # within 0.5% of previous close
        
        if position == 0:
            # Long: breakout above R1 + volume filter + bullish week
            if breakout_up and volume_filter and weekly_bullish:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + bearish week
            elif breakout_down and volume_filter and weekly_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to previous close or break below S1
            if return_to_prev_close or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to previous close or break above R1
            if return_to_prev_close or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0