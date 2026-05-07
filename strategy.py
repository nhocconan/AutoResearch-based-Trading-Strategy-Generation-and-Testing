#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction (structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly high/low/close for pivot
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point (classic)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility (ATR > 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high, above weekly pivot (bullish bias), above EMA50 (uptrend), with volume spike
            buffer = 0.001 * close[i]  # 0.1% buffer to avoid whipsaws
            if (close[i] > donchian_high[i] + buffer and 
                close[i] > weekly_pivot_aligned[i] + buffer and   # above weekly pivot
                close[i] > ema_50_1d_aligned[i] + buffer and   # uptrend
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, below weekly pivot (bearish bias), below EMA50 (downtrend), with volume spike
            elif (close[i] < donchian_low[i] - buffer and 
                  close[i] < weekly_pivot_aligned[i] - buffer and   # below weekly pivot
                  close[i] < ema_50_1d_aligned[i] - buffer and   # downtrend
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to weekly pivot (mean reversion to weekly bias)
            if position == 1:  # Long exit
                if close[i] < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Short exit
                if close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals