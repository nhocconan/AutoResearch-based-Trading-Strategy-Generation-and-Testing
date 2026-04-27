#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_HTF_v2
Hypothesis: 6h Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation.
Donchian channels provide clear breakout levels, 1d EMA50 filters counter-trend moves,
volume spike confirms breakout strength. Designed for low trade frequency (12-37/year)
to minimize fee drag. Works in both bull and bear markets by aligning with daily trend.
Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Donchian(20) channels
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for Donchian(20) and volume average
    start_idx = max(100, donchian_window, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 1d trend with volume spike
            # Long: price breaks above upper channel AND 1d trend is up (close > EMA50) AND volume spike
            # Short: price breaks below lower channel AND 1d trend is down (close < EMA50) AND volume spike
            long_breakout = close_val > upper[i]
            short_breakout = close_val < lower[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below lower channel (failed breakout) or trend reverses
            if close_val < lower[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above upper channel (failed breakout) or trend reverses
            if close_val > upper[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_HTF_v2"
timeframe = "6h"
leverage = 1.0