#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action around daily pivot points with volume confirmation
# and weekly trend filter. Long when price breaks above R3 with volume in uptrend,
# short when breaks below S3 with volume in downtrend. Uses pivot points as
# institutional support/resistance levels. Designed for 50-150 total trades over 4 years.

name = "exp_13332_12h_pivot_breakout_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20  # Lookback for pivot calculation (20 days)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivots(high, low, close):
    """Calculate standard pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily pivot points (using lookback period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize pivot arrays
    r3 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    
    # Calculate pivots for each day using rolling window
    for i in range(PIVOT_LOOKBACK - 1, len(close_1d)):
        idx_start = i - PIVOT_LOOKBACK + 1
        idx_end = i + 1
        # Use the last day in the window for pivot calculation
        h = high_1d[idx_end-1]
        l = low_1d[idx_end-1]
        c = close_1d[idx_end-1]
        _, _, _, r3_val, _, _, s3_val = calculate_pivots(h, l, c)
        r3[idx_end-1] = r3_val
        s3[idx_end-1] = s3_val
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, 20)  # 20-period EMA on weekly
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Align daily R3/S3 to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals using daily R3/S3
        breakout_up = volume_ok and uptrend and (high[i] > r3_aligned[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < s3_aligned[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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