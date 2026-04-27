#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout (20-period) with 1d EMA50 trend filter and volume confirmation.
Breakouts occur when price moves beyond 4h Donchian channels (20-period high/low), filtered by daily EMA50 trend.
Volume > 1.5x average confirms breakout strength. Uses discrete position size (±0.20) to minimize fee churn.
Target: 15-37 trades/year (60-150 over 4 years). Uses 4h/1d for signal direction, 1h only for entry timing.
Includes session filter (08-20 UTC) to reduce noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    period = 20
    for i in range(period-1, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-period+1:i+1])
        donchian_low[i] = np.min(low_4h[i-period+1:i+1])
    
    # Align Donchian channels to 1h timeframe (waits for 4h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate EMA50 on 1d data for trend filter
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align EMA50 to 1h timeframe (waits for 1d bar close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation on 1h data
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Warmup: need Donchian (20), EMA50 (50), volume MA (20)
    start_idx = max(period-1, 49, vol_ma_period) + 1  # +1 for alignment delay
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below EMA50
        uptrend = price > ema_50_aligned[i]
        downtrend = price < ema_50_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high in uptrend with volume
            if uptrend and price > donchian_high_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below 4h Donchian low in downtrend with volume
            elif downtrend and price < donchian_low_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or trend changes
            if price < donchian_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or trend changes
            if price > donchian_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian20_4hBreakout_1dEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0