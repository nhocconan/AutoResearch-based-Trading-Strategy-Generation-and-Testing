#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14057_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA and volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 and 20-period volume average
    ema_50 = calculate_ema(close_1d, 50)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 4h data for Donchian channels and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss (2 * ATR approximation using 20-period range)
        if position != 0:
            range_20 = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values[i]
            if position == 1 and close[i] <= entry_price - 2.0 * range_20:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and close[i] >= entry_price + 2.0 * range_20:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_avg_20_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: price breaks above Donchian high, above EMA50, with volume
            if close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, below EMA50, with volume
            elif close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below EMA50
            if close[i] < donch_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or above EMA50
            if close[i] > donch_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals