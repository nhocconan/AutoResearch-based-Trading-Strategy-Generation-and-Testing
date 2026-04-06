# 4HOUR CAMARILLA PIVOT + VOLUME CONFIRMATION + 1D TREND FILTER
# Hypothesis: 1D trend filters false breakouts, volume confirms institutional interest, Camarilla levels provide high-probability reversal/breakout zones. Works in bull/bear via trend filter.
# Target: 100-150 trades over 4 years (25-38/year) to stay under 400 hard max.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12740_4h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
PIVOT_PERIOD = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2
TREND_EMA_FAST = 9
TREND_EMA_SLOW = 21

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

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    return r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily trend filter
    close_1d = df_1d['close'].values
    ema_fast_1d = calculate_ema(close_1d, TREND_EMA_FAST)
    ema_slow_1d = calculate_ema(close_1d, TREND_EMA_SLOW)
    trend_up_1d = ema_fast_1d > ema_slow_1d
    trend_down_1d = ema_fast_1d < ema_slow_1d
    
    # Align daily trend to 4h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r3_1d, s3_1d, r4_1d, s4_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 4h indicators
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
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, TREND_EMA_SLOW) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(trend_up_1d_aligned[i]):
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
        
        # Fade at R3/S3, breakout at R4/S4 (only in trend direction)
        fade_long = volume_ok and close[i] <= s3_1d_aligned[i] and trend_up_1d_aligned[i]
        fade_short = volume_ok and close[i] >= r3_1d_aligned[i] and trend_down_1d_aligned[i]
        breakout_long = volume_ok and close[i] >= r4_1d_aligned[i] and trend_up_1d_aligned[i]
        breakout_short = volume_ok and close[i] <= s4_1d_aligned[i] and trend_down_1d_aligned[i]
        
        # Entry conditions
        long_entry = fade_long or breakout_long
        short_entry = fade_short or breakout_short
        
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