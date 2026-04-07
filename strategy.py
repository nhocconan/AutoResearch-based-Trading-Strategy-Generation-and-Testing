#!/usr/bin/env python3
"""
6h_donchian_breakout_weekly_pivot_volume_v1
Hypothesis: On 6h timeframe, use Donchian channel breakouts (20-period) with weekly pivot direction filter and volume confirmation. Enter long when price breaks above upper Donchian band with price above weekly pivot and volume confirmation; enter short when price breaks below lower Donchian band with price below weekly pivot and volume confirmation. Exit when price returns to the middle of the Donchian channel. Weekly pivot provides long-term trend bias to avoid counter-trend trades. Volume confirmation ensures institutional participation. Designed for 6h timeframe to achieve 12-37 trades/year with low turnover and high edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (trend filter)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivot)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # We only need the pivot level for trend filter, not S1/R1
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Donchian channel (20-period) on 6h timeframe
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    # Middle band = (upper + lower) / 2
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper = high_series.rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = low_series.rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i]) or np.isnan(weekly_pivot_6h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit when price returns to middle of Donchian channel
            if close[i] <= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to middle of Donchian channel
            if close[i] >= donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band
            # with price above weekly pivot (uptrend bias) and volume confirmation
            long_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i]
            above_weekly_pivot = close[i] > weekly_pivot_6h[i]
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                position = 1
                signals[i] = 0.25
            
            # Short entry: price breaks below lower Donchian band
            # with price below weekly pivot (downtrend bias) and volume confirmation
            short_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i]
            below_weekly_pivot = close[i] < weekly_pivot_6h[i]
            
            if short_breakout and below_weekly_pivot and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals