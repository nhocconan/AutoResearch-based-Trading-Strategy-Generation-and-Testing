#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12787_6d_weekly_pivot_breakout"
timeframe = "6h"
leverage = 1.0

# Parameters
WEEKLY_LOOKBACK = 5  # weeks for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (standard formula)"""
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
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high/low/close from daily data
    # Group by week (Monday to Friday)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate weekly values using 5-day rolling window (approximation)
    weekly_high = pd.Series(daily_high).rolling(window=WEEKLY_LOOKBACK*5, min_periods=WEEKLY_LOOKBACK*5).max().values
    weekly_low = pd.Series(daily_low).rolling(window=WEEKLY_LOOKBACK*5, min_periods=WEEKLY_LOOKBACK*5).min().values
    weekly_close = pd.Series(daily_close).rolling(window=WEEKLY_LOOKBACK*5, min_periods=WEEKLY_LOOKBACK*5).last().values
    
    # Calculate weekly pivot points
    pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r2 + (r2 - s2))  # R4 = R3 + (R2 - S2)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s2 - (r2 - s2))  # S4 = S3 - (R2 - S2)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_LOOKBACK*5, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot levels not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Breakout above R4 or breakdown below S4
        breakout_long = volume_ok and close[i] >= r4_aligned[i]
        breakout_short = volume_ok and close[i] <= s4_aligned[i]
        
        # Fade at R3/S3 with volume confirmation
        fade_long = volume_ok and close[i] <= r3_aligned[i] and close[i] > s3_aligned[i]
        fade_short = volume_ok and close[i] >= s3_aligned[i] and close[i] < r3_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_long and close[i] < weekly_close[i]:  # Fade long only if below weekly close
                signals[i] = -SIGNAL_SIZE * 0.5  # Half size for fade
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short and close[i] > weekly_close[i]:  # Fade short only if above weekly close
                signals[i] = SIGNAL_SIZE * 0.5  # Half size for fade
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals