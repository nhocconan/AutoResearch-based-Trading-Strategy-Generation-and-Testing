#!/usr/bin/env python3
"""
Experiment #10931: 6h Camarilla Pivot Reversal with 1d Trend and Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3) act as strong reversal zones in ranging markets.
Price rejecting these levels with volume confirmation provides high-probability entries.
In trending markets (1d trend), we take breakouts at R4/S4. Works in both bull and bear:
- Ranges: fade at R3/S3 (mean reversion)
- Trends: breakout continuation at R4/S4 (trend following)
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10931_6h_camarilla_reversal_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
DAILY_EMA_PERIOD = 21

def calculate_pivots(high, low, close):
    """Calculate standard pivot point"""
    return (high + low + close) / 3.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = calculate_pivots(high, low, close)
    range_val = high - low
    return {
        'R4': pivot + (range_val * CAMARILLA_MULT * 1.5),
        'R3': pivot + (range_val * CAMARILLA_MULT * 1.0),
        'R2': pivot + (range_val * CAMARILLA_MULT * 0.5),
        'R1': pivot + (range_val * CAMARILLA_MULT * 0.25),
        'S1': pivot - (range_val * CAMARILLA_MULT * 0.25),
        'S2': pivot - (range_val * CAMARILLA_MULT * 0.5),
        'S3': pivot - (range_val * CAMARILLA_MULT * 1.0),
        'S4': pivot - (range_val * CAMARILLA_MULT * 1.5)
    }

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily indicators
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_levels = []
    for i in range(len(daily_high)):
        levels = calculate_camarilla(daily_high[i], daily_low[i], daily_close[i])
        camarilla_levels.append(levels)
    
    # Extract arrays for each level
    r4 = np.array([l['R4'] for l in camarilla_levels])
    r3 = np.array([l['R3'] for l in camarilla_levels])
    s3 = np.array([l['S3'] for l in camarilla_levels])
    s4 = np.array([l['S4'] for l in camarilla_levels])
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Daily EMA for trend filter
    ema_daily = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
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
    start = max(PIVOT_LOOKBACK + 1, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(r3_aligned[i]) or np.isnan(ema_daily_aligned[i]):
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
        
        # Get Camarilla levels (using previous day's levels)
        idx = i - 1
        if idx < 0 or np.isnan(r3_aligned[idx]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (daily)
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Price action
        price = close[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if volume_ok:
            # Ranging market: fade at R3/S3
            if not uptrend_daily and not downtrend_daily:  # sideways market
                # Long when price rejects S3 with volume
                if low[i] <= s3_aligned[idx] and close[i] > s3_aligned[idx]:
                    long_entry = True
                # Short when price rejects R3 with volume
                if high[i] >= r3_aligned[idx] and close[i] < r3_aligned[idx]:
                    short_entry = True
            # Trending market: breakout continuation at R4/S4
            else:
                # Long breakout in uptrend
                if uptrend_daily and high[i] > r4_aligned[idx]:
                    long_entry = True
                # Short breakdown in downtrend
                if downtrend_daily and low[i] < s4_aligned[idx]:
                    short_entry = True
        
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