#!/usr/bin/env python3
name = "1d_WeeklyTrend_VolumeBreakout"
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
    
    # Weekly trend: price above/below 20-week EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    trend_up = close > ema_20w_aligned
    
    # Daily volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # Daily price breakout: Donchian(20) breakout
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA, Donchian, volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: avoid counter-trend trades in weak trends
        if not trend_up[i] and not (close[i] < ema_20w_aligned[i]):
            # In weak weekly downtrend, only allow shorts
            if position == 1:
                signals[i] = 0.0
                position = 0
            continue
        if trend_up[i] and not (close[i] > ema_20w_aligned[i]):
            # In weak weekly uptrend, only allow longs
            if position == -1:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + weekly uptrend + volume filter
            if close[i] > donch_upper[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + weekly downtrend + volume filter
            elif close[i] < donch_lower[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown or weekly trend down
            if close[i] < donch_lower[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up or weekly trend up
            if close[i] > donch_upper[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals