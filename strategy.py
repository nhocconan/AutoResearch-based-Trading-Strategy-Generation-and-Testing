#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + weekly trend filter + volume confirmation
# Uses weekly Donchian(20) to define long-term trend and 1d Donchian(20) for entries
# Only trades in direction of weekly trend with volume confirmation
# Designed for low trade frequency (<25/year) to minimize fee drag
# Works in bull markets (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
name = "1d_DonchianBreakout_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Donchian(20) for trend direction
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    weekly_donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_high)
    weekly_donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donchian_low)
    
    # 1d Donchian(20) for entry signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_donchian_high_aligned[i]) or np.isnan(weekly_donchian_low_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine weekly trend
        # Uptrend: price above weekly Donchian high
        # Downtrend: price below weekly Donchian low
        # Ranging: between the two
        weekly_uptrend = price > weekly_donchian_high_aligned[i]
        weekly_downtrend = price < weekly_donchian_low_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: weekly uptrend + price breaks above 1d Donchian high + volume
            if weekly_uptrend and price > donchian_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below 1d Donchian low + volume
            elif weekly_downtrend and price < donchian_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 1d Donchian low or weekly trend changes
            if price < donchian_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above 1d Donchian high or weekly trend changes
            if price > donchian_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals