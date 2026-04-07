#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout + 1d EMA(50) trend + Volume filter
# Hypothesis: Breakout trading with trend filter and volume confirmation works in both bull and bear markets.
# Donchian channels capture breakouts, EMA(50) filters direction, volume avoids false breakouts.
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.

name = "4h_donchian20_volume_1d_ema_trend_v1"
timeframe = "4h"
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
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # 4h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h Volume filter: volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band or trend changes
            if low[i] <= low_roll[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band or trend changes
            if high[i] >= high_roll[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily EMA trend with volume confirmation
            if vol_ok:
                if close[i] > ema_50_4h[i]:  # Uptrend
                    if high[i] >= high_roll[i]:  # Breakout above upper band
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] <= low_roll[i]:  # Breakdown below lower band
                        position = -1
                        signals[i] = -0.25
    
    return signals