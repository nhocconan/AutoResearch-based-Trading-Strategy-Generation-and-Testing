#!/usr/bin/env python3
"""
Experiment #10291: 6h Donchian Breakout + Daily Pivot Direction + Volume Spike
Hypothesis: Donchian(20) breakouts in the direction of daily pivot trend (S5/S4 for downtrend, S3/S2 for uptrend)
with volume confirmation provide high-probability trend continuation. Works in bull/bear via pivot-based trend filter.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10291_6h_donchian_breakout_daily_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
PIVOT_PERIOD = 1  # daily pivot
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P=(H+L+C)/3, S1=2P-H, S2=P-(H-L), S3=L-2(H-P), S4=S3-(H-L), S5=S4-(H-L)"""
    pivot = (high + low + close) / 3.0
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    s4 = s3 - (high - low)
    s5 = s4 - (high - low)
    return pivot, s1, s2, s3, s4, s5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    _, _, s2, s3, s4, s5 = calculate_pivot_points(daily_high, daily_low, daily_close)
    
    # Determine trend based on S4/S5 (downtrend) vs S2/S3 (uptrend)
    # Uptrend: close > S3, Downtrend: close < S4
    daily_uptrend = daily_close > s3
    daily_downtrend = daily_close < s4
    
    # Align trend indicators to 6h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_daily, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_daily, daily_downtrend.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily trend not available
        if np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: daily uptrend/downtrend
        is_uptrend = daily_uptrend_aligned[i] > 0.5
        is_downtrend = daily_downtrend_aligned[i] > 0.5
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of daily trend with volume
        long_entry = bullish_breakout and is_uptrend and volume_spike
        short_entry = bearish_breakout and is_downtrend and volume_spike
        
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
</x>