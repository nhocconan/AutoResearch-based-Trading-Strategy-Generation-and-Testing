#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout with Volume and Weekly Trend Filter
# Hypothesis: Trade daily Donchian(20) breakouts in direction of weekly EMA(20) trend
# with volume confirmation. Works in bull/bear by following weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_donchian20_volume_weekly_trend_v1"
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
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily Donchian(20) breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: daily volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if low[i] <= low_20[i] or close[i] < ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if high[i] >= high_20[i] or close[i] > ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of weekly EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_weekly_aligned[i]:  # Uptrend
                    if high[i] >= high_20[i] and close[i] > high_20[i]:  # Breakout above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] <= low_20[i] and close[i] < low_20[i]:  # Breakdown below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals