#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d trend filter and session filter (08-20 UTC)
# 4h Donchian(20) provides major support/resistance levels that work in both bull and bear markets
# 1d EMA(50) filter ensures we trade with the higher timeframe trend (avoid counter-trend whipsaws)
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_donchian_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    dm_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or
            np.isnan(dm_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit on retracement to Donchian midpoint
            if close[i] < dm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit on retracement to Donchian midpoint
            if close[i] > dm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout trading with 1d trend filter
            # Long on Donchian high breakout when 1d EMA(50) is rising
            # Short on Donchian low breakout when 1d EMA(50) is falling
            if i >= 1:  # Need previous EMA to determine trend
                ema_rising = ema_50_aligned[i] > ema_50_aligned[i-1]
                ema_falling = ema_50_aligned[i] < ema_50_aligned[i-1]
                
                if close[i] > dh_aligned[i] and ema_rising:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < dl_aligned[i] and ema_falling:
                    position = -1
                    signals[i] = -0.20
    
    return signals