#!/usr/bin/env python3
# 12h_donchian_breakout_1w_trend_volume_v1
# Hypothesis: Use weekly Donchian breakouts (20-period) with 1d trend filter and volume confirmation.
# Long when price breaks above weekly Donchian high with uptrend; short when breaks below weekly Donchian low with downtrend.
# Weekly timeframe provides institutional support/resistance that works in both bull and bear markets.
# Volume filter ensures breakout conviction. Trend filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Donchian high: highest high of last 20 weeks
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low of last 20 weeks
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 28-period average (14 days)
    vol_period = 28
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(20, 50, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout entries with trend and volume confirmation
            # Long breakout when price breaks above weekly Donchian high with uptrend
            if (close[i] > donchian_high_aligned[i] and close[i] > ema_daily_aligned[i] and
                volume_filter):
                position = 1
                signals[i] = 0.25
            # Short breakdown when price breaks below weekly Donchian low with downtrend
            elif (close[i] < donchian_low_aligned[i] and close[i] < ema_daily_aligned[i] and
                  volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals