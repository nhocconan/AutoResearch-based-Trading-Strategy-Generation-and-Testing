#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 1d EMA Trend + Volume
# Hypothesis: Breakouts of 4h Donchian(20) in direction of daily EMA(20) trend
# with volume confirmation work in both bull/bear markets by following trend.
# Target: 20-50 total trades per symbol over 4 years (5-12.5/year) to minimize fee drag.

name = "4h_donchian_breakout_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily EMA(20) for trend filter
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_4h[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price touches 4h low or trend changes
            if low[i] <= low_min[i] or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches 4h high or trend changes
            if high[i] >= high_max[i] or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_4h[i]:  # Uptrend
                    if high[i] >= high_max[i] and close[i] > high_max[i]:
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] <= low_min[i] and close[i] < low_min[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals