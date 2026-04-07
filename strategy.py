#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily 20-period Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Daily Donchian(20) breakouts aligned with weekly trend (price > weekly SMA50) and volume > 1.5x average capture sustained moves in both bull and bear markets.
# Weekly trend filter reduces whipsaws, volume confirmation ensures breakout strength.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "daily_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly SMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_close_series = pd.Series(weekly_close)
    weekly_sma50 = weekly_close_series.rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA50 to daily
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma50)
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(weekly_sma50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < weekly_sma50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > weekly_sma50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Weekly trend filter: price above weekly SMA50 for long, below for short
            if close[i] > weekly_sma50_aligned[i]:
                # Long entry: breakout above Donchian high with volume
                if high[i] > donchian_high[i] and close[i] > donchian_high[i] and vol_filter[i]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < weekly_sma50_aligned[i]:
                # Short entry: breakdown below Donchian low with volume
                if low[i] < donchian_low[i] and close[i] < donchian_low[i] and vol_filter[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals