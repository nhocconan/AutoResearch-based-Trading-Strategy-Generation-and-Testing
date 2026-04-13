#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d weekly pivot bias and volume confirmation.
    # Weekly pivot from 1d data determines structural bias: price above weekly pivot = bullish bias (long breakouts only),
    # price below weekly pivot = bearish bias (short breakouts only). Camarilla levels (H3/L3) from 1d provide entry zones.
    # Volume confirmation ensures breakout validity. Target: 50-150 total trades over 4 years = 12-37/year.
    # Works in bull markets (long breakouts with bullish bias) and bear markets (short breakouts with bearish bias).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and Camarilla levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d OHLC
    # Weekly pivot = (Prior week HIGH + LOW + CLOSE) / 3
    # We approximate weekly using rolling window of 5 days (1 trading week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day rolling window
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    
    # Align weekly pivot to 12h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate 1d Camarilla levels (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma[i]
        
        # Breakout conditions at Camarilla H3/L3 levels
        long_breakout = close[i] > camarilla_h3_aligned[i]  # Break above H3
        short_breakout = close[i] < camarilla_l3_aligned[i]  # Break below L3
        
        # Weekly pivot bias: price above pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Entry conditions: breakout in direction of weekly pivot bias
        long_entry = long_breakout and bullish_bias and volume_filter
        short_entry = short_breakout and bearish_bias and volume_filter
        
        # Exit conditions: opposite breakout or loss of bias
        long_exit = short_breakout or not bullish_bias
        short_exit = long_breakout or not bearish_bias
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_weekly_pivot_v1"
timeframe = "12h"
leverage = 1.0