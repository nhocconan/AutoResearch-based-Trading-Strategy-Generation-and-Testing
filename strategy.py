#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout + 1d Volume Confirmation + Trend Filter
# Hypothesis: Breakouts from Donchian(20) channels with volume confirmation
# in the direction of the daily EMA(20) trend. Works in bull/bear by trading
# with the daily trend. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_donchian_breakout_1d_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily volume (use previous day's volume to avoid look-ahead)
    vol_daily = df_daily['volume'].values
    vol_daily_prev = np.roll(vol_daily, 1)
    vol_daily_prev[0] = np.nan
    vol_daily_6h = align_htf_to_ltf(prices, df_daily, vol_daily_prev)
    
    # Daily EMA(20) for trend filter (use previous day's EMA)
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily_prev = np.roll(ema_20_daily, 1)
    ema_20_daily_prev[0] = np.nan
    ema_20_6h = align_htf_to_ltf(prices, df_daily, ema_20_daily_prev)
    
    # 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(vol_daily_6h[i]) or np.isnan(ema_20_6h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x previous day's average volume
        vol_ok = volume[i] > (vol_daily_6h[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price touches Donchian low or trend changes
            if low[i] <= donchian_low[i] or close[i] < ema_20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches Donchian high or trend changes
            if high[i] >= donchian_high[i] or close[i] > ema_20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily EMA trend with volume confirmation
            if vol_ok:
                if close[i] > ema_20_6h[i]:  # Uptrend
                    if high[i] > donchian_high[i-1]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < donchian_low[i-1]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals