#!/usr/bin/env python3
"""
4h_Donchian20_12hTrend_VolumeConfirm_v1
Hypothesis: Donchian channel breakouts on 4h with 12h trend filter and volume confirmation
work in both bull and bear markets by capturing strong moves while avoiding whipsaws.
The 12h trend filter ensures we only trade in the direction of the higher timeframe trend,
reducing false signals. Volume confirmation ensures breakouts have institutional backing.
Target: 20-40 trades/year to minimize fee drag on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels: upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_12h = calculate_ema(df_12h['close'].values, 50)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels on 4h
    upper, lower = calculate_donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian, EMA, and volume MA
    start_idx = max(20, 50)  # Donchian needs 20, EMA needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_12h_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band, 12h trend up (close > EMA), volume confirmation
            if close[i] > upper[i] and close[i] > ema_trend and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian band, 12h trend down (close < EMA), volume confirmation
            elif close[i] < lower[i] and close[i] < ema_trend and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian band or 12h trend turns down
            if close[i] < lower[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper Donchian band or 12h trend turns up
            if close[i] > upper[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_12hTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0