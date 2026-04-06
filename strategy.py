#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12859_6h_12h_1d_triple_ema_slope_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_FAST_PERIOD = 8
EMA_MED_PERIOD = 21
EMA_SLOW_PERIOD = 55
SLOPE_THRESHOLD = 0.001
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # Max 4 days (24 * 6h)

def calculate_slope(values):
    """Calculate linear regression slope over last 3 points"""
    if len(values) < 3:
        return 0.0
    x = np.array([0, 1, 2])
    y = values[-3:]
    slope = np.polyfit(x, y, 1)[0]
    return slope

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs on 12h
    close_12h = df_12h['close'].values
    ema_fast_12h = pd.Series(close_12h).ewm(span=EMA_FAST_PERIOD, adjust=False).values
    ema_med_12h = pd.Series(close_12h).ewm(span=EMA_MED_PERIOD, adjust=False).values
    ema_slow_12h = pd.Series(close_12h).ewm(span=EMA_SLOW_PERIOD, adjust=False).values
    
    # Calculate EMAs on 1d
    close_1d = df_1d['close'].values
    ema_fast_1d = pd.Series(close_1d).ewm(span=EMA_FAST_PERIOD, adjust=False).values
    ema_med_1d = pd.Series(close_1d).ewm(span=EMA_MED_PERIOD, adjust=False).values
    ema_slow_1d = pd.Series(close_1d).ewm(span=EMA_SLOW_PERIOD, adjust=False).values
    
    # Align to 6h timeframe
    ema_fast_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_fast_12h)
    ema_med_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_med_12h)
    ema_slow_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slow_12h)
    ema_fast_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_fast_1d)
    ema_med_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_med_1d)
    ema_slow_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slow_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    atr = pd.Series(high - low).rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    atr = pd.Series(atr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA_SLOW_PERIOD, ATR_PERIOD) + 5
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if EMA data not available
        if np.isnan(ema_fast_12h_aligned[i]) or np.isnan(ema_med_12h_aligned[i]) or np.isnan(ema_slow_12h_aligned[i]) or \
           np.isnan(ema_fast_1d_aligned[i]) or np.isnan(ema_med_1d_aligned[i]) or np.isnan(ema_slow_1d_aligned[i]):
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
        
        # Calculate EMA slopes for 12h and 1d
        fast_12h_slope = calculate_slope(ema_fast_12h_aligned[max(0, i-4):i+1])
        med_12h_slope = calculate_slope(ema_med_12h_aligned[max(0, i-4):i+1])
        slow_12h_slope = calculate_slope(ema_slow_12h_aligned[max(0, i-4):i+1])
        fast_1d_slope = calculate_slope(ema_fast_1d_aligned[max(0, i-4):i+1])
        med_1d_slope = calculate_slope(ema_med_1d_aligned[max(0, i-4):i+1])
        slow_1d_slope = calculate_slope(ema_slow_1d_aligned[max(0, i-4):i+1])
        
        # Bullish alignment: fast > med > slow AND positive slopes
        bullish_12h = (ema_fast_12h_aligned[i] > ema_med_12h_aligned[i] > ema_slow_12h_aligned[i]) and \
                      (fast_12h_slope > SLOPE_THRESHOLD and med_12h_slope > SLOPE_THRESHOLD)
        bullish_1d = (ema_fast_1d_aligned[i] > ema_med_1d_aligned[i] > ema_slow_1d_aligned[i]) and \
                     (fast_1d_slope > SLOPE_THRESHOLD and med_1d_slope > SLOPE_THRESHOLD)
        
        # Bearish alignment: fast < med < slow AND negative slopes
        bearish_12h = (ema_fast_12h_aligned[i] < ema_med_12h_aligned[i] < ema_slow_12h_aligned[i]) and \
                      (fast_12h_slope < -SLOPE_THRESHOLD and med_12h_slope < -SLOPE_THRESHOLD)
        bearish_1d = (ema_fast_1d_aligned[i] < ema_med_1d_aligned[i] < ema_slow_1d_aligned[i]) and \
                     (fast_1d_slope < -SLOPE_THRESHOLD and med_1d_slope < -SLOPE_THRESHOLD)
        
        # Generate signals
        if position == 0:
            if bullish_12h and bullish_1d:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif bearish_12h and bearish_1d:
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