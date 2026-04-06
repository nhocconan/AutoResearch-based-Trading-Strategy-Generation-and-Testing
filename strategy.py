#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Monthly KAMA trend with weekly pivot confirmation on 1d timeframe
# Works in bull/bear because KAMA adapts to volatility (fast in trends, slow in ranges),
# while weekly pivots provide structural support/resistance. Volume filters false breakouts.
# Target: 40-80 trades over 4 years (10-20/year) to minimize fee drag.

name = "exp_12944_1d_kama_monthly_pivot_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_FAST = 2
KAMA_ER_SLOW = 30
KAMA_PERIOD = 10
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_kama(close, er_fast, er_slow, period):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, period))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, prepend=close[0])).sum()
    # Use pandas for proper rolling calculation
    close_s = pd.Series(close)
    change_s = pd.Series(np.abs(close_s - close_s.shift(period)))
    volatility_s = pd.Series(np.abs(close_s.diff())).rolling(window=period, min_periods=1).sum()
    
    er = change_s / volatility_s.replace(0, np.finfo(float).eps)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_pivot_points(high, low, close):
    """Calculate monthly pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

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
    
    # Load monthly data ONCE before loop
    df_monthly = get_htf_data(prices, '1M')
    
    # Calculate monthly KAMA
    close_m = df_monthly['close'].values
    kama_vals = calculate_kama(close_m, KAMA_ER_FAST, KAMA_ER_SLOW, KAMA_PERIOD)
    
    # Align to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_monthly, kama_vals)
    
    # Load weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot_vals = np.zeros(len(close_w))
    r1_vals = np.zeros(len(close_w))
    r2_vals = np.zeros(len(close_w))
    r3_vals = np.zeros(len(close_w))
    s1_vals = np.zeros(len(close_w))
    s2_vals = np.zeros(len(close_w))
    s3_vals = np.zeros(len(close_w))
    
    for i in range(len(close_w)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_w[i], low_w[i], close_w[i])
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
    
    # Align weekly pivots to daily
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_vals)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_vals)
    
    # Calculate daily indicators
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
    start = max(KAMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if KAMA or pivot levels not available
        if np.isnan(kama_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # KAMA trend direction: price above/below KAMA
        kama_long = close[i] > kama_aligned[i]
        kama_short = close[i] < kama_aligned[i]
        
        # Breakout above R3 or breakdown below S3 with volume
        breakout_long = volume_ok and kama_long and close[i] >= r3_aligned[i]
        breakout_short = volume_ok and kama_short and close[i] <= s3_aligned[i]
        
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
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals