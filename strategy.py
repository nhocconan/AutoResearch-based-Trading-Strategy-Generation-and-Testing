#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1D Donchian Breakout + Weekly EMA Trend + Volume Confirmation
# Hypothesis: Buy breakouts above 20-day high in weekly uptrend, sell breakdowns below 20-day low in weekly downtrend.
# Uses weekly EMA(20) as trend filter to align with higher timeframe direction, reducing whipsaws in sideways markets.
# Volume confirmation ensures breakouts have conviction. Works in bull/bear by trading with weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_donchian_breakout_weekly_ema_volume_v2"
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
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: daily volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or trend changes
            if low[i] < low_20[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or trend changes
            if high[i] > high_20[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of weekly EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_aligned[i]:  # Weekly uptrend
                    if high[i] > high_20[i]:  # Breakout above 20-day high
                        position = 1
                        signals[i] = 0.25
                else:  # Weekly downtrend
                    if low[i] < low_20[i]:  # Breakdown below 20-day low
                        position = -1
                        signals[i] = -0.25
    
    return signals