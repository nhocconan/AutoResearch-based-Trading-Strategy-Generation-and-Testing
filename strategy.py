#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA filter.
# Works in bull (breakouts continue) and bear (fades at resistance/support).
# Volume filter reduces false breakouts. Target: 100-200 trades over 4 years.

name = "exp_14061_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d
    ema_50_1d = calculate_ema(close_1d, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 4h
    upper, lower = calculate_donchian(high, low, 20)
    
    # Calculate 20-period average volume on 4h
    avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA)
    start = max(20, 50)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * avg_vol[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume, above 1d EMA
            if close[i] > upper[i] and vol_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band with volume, below 1d EMA
            elif close[i] < lower[i] and vol_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band or below 1d EMA
            if close[i] < lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper band or above 1d EMA
            if close[i] > upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals