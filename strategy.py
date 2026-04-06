#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h weekly pivot point reversal strategy. Uses weekly pivot levels (R3/S3) for reversal entries
# in ranging markets and (R4/S4) for breakout continuation in trending markets. Combines with volume
# confirmation to filter false signals. Designed for 60-120 total trades over 4 years (15-30/year).
# Works in bull markets (breakouts at R4/S4 with volume) and bear markets (reversals at R3/S3 with volume).

name = "exp_13755_6h_weekly_pivot_reversal_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # Minimum bars to confirm pivot level respect
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, R2, R3, R4, S1, S2, S3, S4"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Initialize pivot arrays
    r3_weekly = np.full_like(close_weekly, np.nan)
    s3_weekly = np.full_like(close_weekly, np.nan)
    r4_weekly = np.full_like(close_weekly, np.nan)
    s4_weekly = np.full_like(close_weekly, np.nan)
    
    # Calculate pivot points for each week
    for i in range(len(close_weekly)):
        _, _, _, r3, r4, _, _, s3, s4 = calculate_pivot_points(
            high_weekly[i], low_weekly[i], close_weekly[i]
        )
        r3_weekly[i] = r3
        s3_weekly[i] = s3
        r4_weekly[i] = r4
        s4_weekly[i] = s4
    
    # Align weekly pivots to 6h timeframe (forward fill with shift(1))
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3_weekly)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3_weekly)
    r4_6h = align_htf_to_ltf(prices, df_weekly, r4_weekly)
    s4_6h = align_htf_to_ltf(prices, df_weekly, s4_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price action
        price = close[i]
        prev_price = close[i-1]
        
        # Reversal signals at R3/S3 (fade extremes)
        reversal_long = volume_ok and price <= s3_6h[i] and prev_price > s3_6h[i]
        reversal_short = volume_ok and price >= r3_6h[i] and prev_price < r3_6h[i]
        
        # Breakout signals at R4/S4 (continuation)
        breakout_long = volume_ok and price >= r4_6h[i] and prev_price < r4_6h[i]
        breakout_short = volume_ok and price <= s4_6h[i] and prev_price > s4_6h[i]
        
        # Generate signals
        if position == 0:
            if reversal_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif reversal_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
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
            # Exit long on reversal at R3 or stop loss
            if price >= r3_6h[i] and prev_price < r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on reversal at S3 or stop loss
            if price <= s3_6h[i] and prev_price > s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals