#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume Filter
# Hypothesis: Donchian(20) breakouts on 12h timeframe in direction of daily trend
# (close > SMA50) with volume confirmation capture momentum moves while avoiding
# whipsaws. Daily trend filter provides robustness across bull/bear markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "12h_daily_donchian_breakout_volume_v1"
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
    
    # Get daily data for trend filter and breakout levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily SMA50 for trend filter
    daily_close = df_daily['close'].values
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    
    # Daily breakout levels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_high_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    daily_low_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 12h timeframe
    sma50_aligned = align_htf_to_ltf(prices, df_daily, daily_sma50)
    high_20_aligned = align_htf_to_ltf(prices, df_daily, daily_high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_daily, daily_low_20)
    
    # Volume filter on 12h: volume > 1.3x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(sma50_aligned[i]) or np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below 20-day low
            if close[i] < low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above 20-day high
            if close[i] > high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Trend filter: price must be on correct side of daily SMA50
            if close[i] > sma50_aligned[i]:
                # Long entry: breakout above 20-day high with volume
                if (high[i] > high_20_aligned[i] and close[i] > high_20_aligned[i] and
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
            else:
                # Short entry: breakdown below 20-day low with volume
                if (low[i] < low_20_aligned[i] and close[i] < low_20_aligned[i] and
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals