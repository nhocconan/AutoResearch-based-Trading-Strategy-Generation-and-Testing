#!/usr/bin/env python3
"""
Experiment #11587: 6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts capture medium-term trends. Weekly pivot levels from 1D provide directional bias, and volume filter ensures institutional participation. Weekly pivots adapt to market regime (bull/bear) by using higher timeframe structure. Target: 100-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_11587_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    return p, r1, s1

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivots from daily data (using last 5 days)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling pivot points (5-day lookback)
    p_pivot = np.full(len(df_1d), np.nan)
    r1_pivot = np.full(len(df_1d), np.nan)
    s1_pivot = np.full(len(df_1d), np.nan)
    
    for i in range(PIVOT_LOOKBACK, len(df_1d)):
        # Use last 5 days including current
        lookback_high = np.max(high_1d[i-PIVOT_LOOKBACK+1:i+1])
        lookback_low = np.min(low_1d[i-PIVOT_LOOKBACK+1:i+1])
        lookback_close = close_1d[i]
        p, r1, s1 = calculate_pivot_points(lookback_high, lookback_low, lookback_close)
        p_pivot[i] = p
        r1_pivot[i] = r1
        s1_pivot[i] = s1
    
    # Align weekly pivots to 6h timeframe
    p_pivot_aligned = align_ltf_to_htf(prices, df_1d, p_pivot)
    r1_pivot_aligned = align_ltf_to_htf(prices, df_1d, r1_pivot)
    s1_pivot_aligned = align_ltf_to_htf(prices, df_1d, s1_pivot)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(p_pivot_aligned[i]) or np.isnan(r1_pivot_aligned[i]) or np.isnan(s1_pivot_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout conditions
        breakout_up = high[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_down = low[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Pivot direction bias (weekly)
        # Price above weekly pivot = bullish bias, below = bearish bias
        pivot_bias_up = close[i] > p_pivot_aligned[i]
        pivot_bias_down = close[i] < p_pivot_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_ok and pivot_bias_up
        short_entry = breakout_down and volume_ok and pivot_bias_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals