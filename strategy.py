#!/usr/bin/env python3
# 12h_breakout_1d_trend_volume_v1
# Hypothesis: Breakout of 12h Donchian(20) channels with daily EMA(50) trend filter and volume confirmation.
# Works in bull markets (breakouts in uptrend) and bear markets (breakouts in downtrend).
# Volume filters out false breakouts. Target: 50-150 total trades over 4 years.

name = "12h_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h
    donch_period = 20
    upper = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Daily EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 24-period average (12 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(donch_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_daily_aligned[i]) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian or trend fails
            if close[i] < lower[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian or trend fails
            if close[i] > upper[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above upper Donchian with uptrend
                if close[i] > upper[i] and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Breakout short: price breaks below lower Donchian with downtrend
                elif close[i] < lower[i] and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals