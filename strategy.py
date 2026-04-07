#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian(20) Breakout + Weekly Trend Filter + Volume Spike
# Hypothesis: Breakout trades in direction of weekly trend with volume confirmation.
# Weekly trend avoids counter-trend trades in chop. Volume ensures momentum.
# Works in bull/bear by following weekly trend. Target: 50-150 total trades (12-37/year).

name = "6h_donchian20_weekly_trend_volume_v1"
timeframe = "6h"
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
    ema_20_6h = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily data for Donchian channels (using 1d for better alignment)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # 20-period Donchian channels from daily
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Upper band: 20-day high
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-day low
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h
    upper_20_6h = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_6h = align_htf_to_ltf(prices, df_daily, lower_20)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price hits lower Donchian or trend changes
            if low[i] <= lower_20_6h[i] or close[i] < ema_20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price hits upper Donchian or trend changes
            if high[i] >= upper_20_6h[i] or close[i] > ema_20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of weekly trend with volume
            if vol_ok:
                if close[i] > ema_20_6h[i]:  # Weekly uptrend
                    if high[i] > upper_20_6h[i]:  # Break above upper Donchian
                        position = 1
                        signals[i] = 0.25
                else:  # Weekly downtrend
                    if low[i] < lower_20_6h[i]:  # Break below lower Donchian
                        position = -1
                        signals[i] = -0.25
    
    return signals