#!/usr/bin/env python3
# 1h_4h1d_donchian_volume_v1
# Hypothesis: 1h strategy using 4h Donchian breakout (20-period) aligned with 1d trend (price above/below 1d EMA50) and volume confirmation (>1.5x 20-period average).
# 4h Donchian provides structural breakout signals; 1d EMA50 acts as trend filter for continuation.
# Volume confirmation ensures institutional participation. Designed for both bull and bear markets by requiring alignment with 1d trend.
# Discrete position sizing (±0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Session filter: 08-20 UTC to reduce noise trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average (1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low OR below 1d EMA50 (trend fails)
            if close[i] < lowest_low_20_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high OR above 1d EMA50 (trend fails)
            if close[i] > highest_high_20_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Need volume confirmation and in-session
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above 4h Donchian high WITH 1d bullish bias
                if close[i] > highest_high_20_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below 4h Donchian low WITH 1d bearish bias
                elif close[i] < lowest_low_20_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals