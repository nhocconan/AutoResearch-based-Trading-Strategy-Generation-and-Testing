#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Daily price breaking above/below 20-day Donchian channel with weekly trend alignment
# and volume confirmation captures trends in both bull and bear markets. Weekly trend filter
# prevents counter-trend trades, reducing whipsaws. Volume confirmation ensures breakout strength.
# Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.

name = "daily_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema50 = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA50 to daily timeframe
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Calculate daily Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if low[i] <= donchian_low[i] or close[i] < weekly_ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if high[i] >= donchian_high[i] or close[i] > weekly_ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume and weekly uptrend
            if high[i] > donchian_high[i] and close[i] > weekly_ema50_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and weekly downtrend
            elif low[i] < donchian_low[i] and close[i] < weekly_ema50_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals