#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using weekly pivot points (R1/S1) with daily EMA(50) trend filter and volume confirmation.
# Uses weekly pivot levels for entry/exit, daily EMA for trend direction, volume for confirmation.
# Designed for ~100 total trades over 4 years (25/year) to avoid excessive fees.
# Works in bull (breakouts above R1 with volume) and bear (breakdowns below S1 with volume) markets.
# Target: 75-150 total trades, 0.25 position size, max DD < -50%.

name = "exp_13748_12h_pivot1w_ema50_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
PIVOT_PERIOD = '1w'  # weekly pivot points
TREND_EMA_PERIOD = 50  # daily EMA for trend
VOLUME_MA_PERIOD = 8   # 12h volume moving average
VOLUME_THRESHOLD = 1.5 # volume must exceed MA by this factor
SIGNAL_SIZE = 0.25     # position size (25% of capital)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivots and daily data for EMA ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Initialize pivot arrays
    pivot_vals = np.full_like(close_weekly, np.nan)
    r1_vals = np.full_like(close_weekly, np.nan)
    s1_vals = np.full_like(close_weekly, np.nan)
    
    # Calculate pivots for each weekly bar
    for i in range(len(close_weekly)):
        pivot, r1, r2, s1, s2 = calculate_pivot_points(high_weekly[i], low_weekly[i], close_weekly[i])
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        s1_vals[i] = s1
    
    # Calculate daily EMA for trend filter
    close_daily = df_daily['close'].values
    ema_daily = calculate_ema(close_daily, TREND_EMA_PERIOD)
    
    # Calculate ATR for stop loss (using 12h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate 12h volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align weekly pivots and daily EMA to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_vals)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_vals)
    ema_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 12h volume)
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from daily EMA
        above_ema = close[i] > ema_aligned[i]
        below_ema = close[i] < ema_aligned[i]
        
        # Pivot breakout signals
        # Long: price breaks above R1 with volume in uptrend
        long_signal = volume_ok and above_ema and close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i]
        # Short: price breaks below S1 with volume in downtrend
        short_signal = volume_ok and below_ema and close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on break below pivot point
            if close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on break above pivot point
            if close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals