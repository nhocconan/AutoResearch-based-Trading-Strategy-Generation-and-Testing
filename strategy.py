#!/usr/bin/env python3
"""
12h_donchian_20_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use Donchian channel breakout (20-period) with 1-week trend filter and volume confirmation.
Enter long when price breaks above 20-period high and 1w trend is up (price > 50-period SMA).
Enter short when price breaks below 20-period low and 1w trend is down (price < 50-period SMA).
Exit when price crosses the 20-period midpoint or trend reverses.
Designed for low trade frequency (12-37/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period SMA on weekly close
    weekly_close = df_1w['close'].values
    weekly_close_s = pd.Series(weekly_close)
    sma50_1w = weekly_close_s.rolling(window=50, min_periods=50).mean().values
    
    # Align to 12h timeframe (shifted by 1 week to avoid look-ahead)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after Donchian and SMA warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter from 1w: up if close > SMA50, down if close < SMA50
        trend_up = close[i] > sma50_1w_aligned[i]
        trend_down = close[i] < sma50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses below midpoint
            if close[i] < donchian_mid[i]:
                exit_long = True
            # Exit on trend reversal (close < SMA50)
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses above midpoint
            if close[i] > donchian_mid[i]:
                exit_short = True
            # Exit on trend reversal (close > SMA50)
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, 1w trend up, volume confirmation
            long_entry = (close[i] > donchian_high[i-1]) and trend_up and vol_confirm
            
            # Short entry: price breaks below Donchian low, 1w trend down, volume confirmation
            short_entry = (close[i] < donchian_low[i-1]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals