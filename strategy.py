#!/usr/bin/env python3
# 12h_donchian_breakout_1w_trend_volume_v1
# Hypothesis: 12h Donchian channel breakout with weekly trend filter and volume confirmation.
# Enter long when price breaks above 12h Donchian upper (20) and weekly EMA trend is up.
# Enter short when price breaks below 12h Donchian lower (20) and weekly EMA trend is down.
# Volume confirms breakout strength. Weekly trend filter ensures alignment with higher timeframe.
# Designed for low-frequency, high-conviction trades to minimize fee drag in both bull/bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # Donchian Channel (20-period) on 12h
    dc_period = 20
    upper = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Weekly EMA trend filter (21-period)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume filter: volume > 1.5x 24-period average (12 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(dc_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_weekly_aligned[i]) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend fails
            if close[i] < lower[i] or close[i] < ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend fails
            if close[i] > upper[i] or close[i] > ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above upper band with uptrend
                if close[i] > upper[i] and close[i] > ema_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below lower band with downtrend
                elif close[i] < lower[i] and close[i] < ema_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals