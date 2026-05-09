#!/usr/bin/env python3
# 1d_Donchian20_Breakout_1wTrend_VolumeSpike
# Hypothesis: Daily Donchian(20) breakouts with weekly EMA50 trend filter and volume spike.
# The weekly EMA50 filters out counter-trend trades, reducing whipsaws in sideways/choppy markets.
# Volume spike ensures breakouts have institutional participation, increasing follow-through.
# Works in bull markets via trend-following breakouts; works in bear markets by avoiding false breakdowns
# in downtrends (only shorts when price < weekly EMA50 and breaks below Donchian low).
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period high/low) from daily data
    # Use pandas for vectorized rolling window with min_periods
    close_ser = pd.Series(close)
    high_ser = pd.Series(high)
    low_ser = pd.Series(low)
    
    donchian_high = high_ser.rolling(window=20, min_periods=20).max().values
    donchian_low = low_ser.rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA50 on weekly close
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        # Initialize with SMA of first 50 periods
        ema_50_1w[49] = np.mean(close_1w[0:50])
        # Calculate EMA for remaining periods
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter: current volume / 20-day average volume
    vol_ser = pd.Series(volume)
    vol_ma = vol_ser.rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.full_like(volume, np.nan)
    # Avoid division by zero
    valid = (vol_ma != 0) & (~np.isnan(vol_ma))
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND uptrend (price > weekly EMA50) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND downtrend (price < weekly EMA50) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend reversal (price < weekly EMA50)
            if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend reversal (price > weekly EMA50)
            if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals