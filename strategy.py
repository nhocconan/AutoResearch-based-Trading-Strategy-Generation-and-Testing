#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Volume
Hypothesis: Weekly Donchian breakouts with volume confirmation and trend filter
capture major trend moves while avoiding whipsaws. Long when price breaks above
20-week high with volume > 1.5x average and price above weekly EMA50. Short when
price breaks below 20-week low with volume confirmation and price below EMA50.
Exit on opposite breakout. Position size: ±0.25. Designed for low trade frequency
(<25/year) to minimize fee drag in both bull and bear markets.
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
    
    # Volume confirmation (20-period MA on daily)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly EMA50 for trend filter
    close_series_weekly = pd.Series(close_weekly)
    ema50_weekly = close_series_weekly.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate weekly rolling high (20 periods) and low (20 periods)
    high_weekly_series = pd.Series(high_weekly)
    low_weekly_series = pd.Series(low_weekly)
    high_20_weekly = high_weekly_series.rolling(window=20, min_periods=20).max().values
    low_20_weekly = low_weekly_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly high/low to daily timeframe
    high_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, high_20_weekly)
    low_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, low_20_weekly)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 50)  # volume MA20, weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(high_20_weekly_aligned[i]) or 
            np.isnan(low_20_weekly_aligned[i]) or 
            np.isnan(ema50_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price > weekly high (20) + volume filter + price above weekly EMA50
            if close[i] > high_20_weekly_aligned[i] and volume_filter and close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly low (20) + volume filter + price below weekly EMA50
            elif close[i] < low_20_weekly_aligned[i] and volume_filter and close[i] < ema50_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < weekly low (20) or price below weekly EMA50
            if close[i] < low_20_weekly_aligned[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > weekly high (20) or price above weekly EMA50
            if close[i] > high_20_weekly_aligned[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0