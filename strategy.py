#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12819_6h_daily_ichimoku_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CONVERSION_PERIOD = 9
BASE_PERIOD = 26
LEADING_SPAN_B_PERIOD = 52
DISPLACEMENT = 26
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 48  # Max 12 days
SIGNAL_SIZE = 0.25

def calculate_ichimoku(high, low):
    """Calculate Ichimoku components"""
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    conversion = (pd.Series(high).rolling(window=CONVERSION_PERIOD, min_periods=CONVERSION_PERIOD).max() + 
                  pd.Series(low).rolling(window=CONVERSION_PERIOD, min_periods=CONVERSION_PERIOD).min()) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    base = (pd.Series(high).rolling(window=BASE_PERIOD, min_periods=BASE_PERIOD).max() + 
            pd.Series(low).rolling(window=BASE_PERIOD, min_periods=BASE_PERIOD).min()) / 2
    
    # Leading Span A (Senkou Span A): (Conversion + Base)/2
    leading_span_a = (conversion + base) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
    leading_span_b = (pd.Series(high).rolling(window=LEADING_SPAN_B_PERIOD, min_periods=LEADING_SPAN_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=LEADING_SPAN_B_PERIOD, min_periods=LEADING_SPAN_B_PERIOD).min()) / 2
    
    return conversion, base, leading_span_a, leading_span_b

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
    if n < 52:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on daily data
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    conversion_d, base_d, leading_span_a_d, leading_span_b_d = calculate_ichimoku(high_d, low_d)
    
    # Align to 6h timeframe
    conversion_a = align_htf_to_ltf(prices, df_daily, conversion_d)
    base_a = align_htf_to_ltf(prices, df_daily, base_d)
    leading_span_a_a = align_htf_to_ltf(prices, df_daily, leading_span_a_d)
    leading_span_b_a = align_htf_to_ltf(prices, df_daily, leading_span_b_d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(LEADING_SPAN_B_PERIOD, ATR_PERIOD) + DISPLACEMENT + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if Ichimoku data not available
        if (np.isnan(conversion_a[i]) or np.isnan(base_a[i]) or 
            np.isnan(leading_span_a_a[i]) or np.isnan(leading_span_b_a[i])):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit to prevent overtrading
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Ichimoku signals
        # Bullish: Conversion > Base AND price above cloud
        bullish = (conversion_a[i] > base_a[i]) and (close[i] > max(leading_span_a_a[i], leading_span_b_a[i]))
        # Bearish: Conversion < Base AND price below cloud
        bearish = (conversion_a[i] < base_a[i]) and (close[i] < min(leading_span_a_a[i], leading_span_b_a[i]))
        
        # Generate signals
        if position == 0:
            if bullish:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif bearish:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals