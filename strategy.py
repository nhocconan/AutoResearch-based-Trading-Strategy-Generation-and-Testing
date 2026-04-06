#!/usr/bin/env python3
"""
Experiment #12279: 6h Donchian Breakout with Daily Pivot and Volume Confirmation
Hypothesis: Combines 6h Donchian(20) breakouts with daily pivot point direction (using R3/S3 levels) and volume confirmation. 
In bull markets: breakouts above R3 with volume continuation. 
In bear markets: breakdowns below S3 with volume confirmation. 
Pivot points act as dynamic support/resistance that adapt to market regime. 
Target: 80-180 total trades over 4 years (20-45/year) with selective entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12279_6h_donchian_daily_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 24  # 4 days of 6h bars
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 0  # Trade all hours for 6h timeframe
SESSION_END_HOUR = 23

def calculate_donchian(high, low, period):
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

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(
        df_daily['high'].values, 
        df_daily['low'].values, 
        df_daily['close'].values
    )
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for look-ahead bias prevention)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Precompute session hours (though we trade all hours for 6h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily pivot data not available (first day)
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Donchian breakout conditions
        donch_breakout_up = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        donch_breakdown_down = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Pivot-based directional bias
        # Long bias: price above R3 (strong bullish)
        # Short bias: price below S3 (strong bearish)
        long_bias = close[i] > r3_aligned[i]
        short_bias = close[i] < s3_aligned[i]
        
        # Entry conditions
        long_entry = volume_ok and donch_breakout_up and long_bias
        short_entry = volume_ok and donch_breakdown_down and short_bias
        
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