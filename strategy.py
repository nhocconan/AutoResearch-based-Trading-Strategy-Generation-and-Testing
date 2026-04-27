#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume spike + session filter
# Donchian breakouts capture momentum moves. 4h EMA50 filters for trend direction.
# Volume spike confirms breakout strength. Session filter (08-20 UTC) reduces noise.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).
# Works in bull/bear: trend filter adapts direction, breakouts catch both breakouts and breakdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above 20-period high + uptrend + volume
        if (close[i] > high_max[i] and 
            close[i] > ema50_4h_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short breakdown: price breaks below 20-period low + downtrend + volume
        elif (close[i] < low_min[i] and 
              close[i] < ema50_4h_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

name = "1h_DonchianBreakout_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0