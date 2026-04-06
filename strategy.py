#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12755_6d_wick_rejection_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WICK_RATIO_PERIOD = 14
WICK_RATIO_THRESHOLD = 0.6  # Body/total range < 0.4 means strong rejection
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MIN_HOLD_BARS = 3  # Minimum holding period to reduce churn

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly trend using close vs SMA50
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_trend = weekly_close > weekly_sma50  # True for uptrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Body-to-range ratio (small body = rejection)
    body = np.abs(close - open_price)
    total_range = high - low
    # Avoid division by zero
    body_ratio = np.where(total_range > 0, body / total_range, 1.0)
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_held = 0
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WICK_RATIO_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, 50) + 1
    
    for i in range(start, n):
        bars_held += 1
        
        # Skip if data not ready
        if np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
                bars_held = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_held = 0
                continue
        
        # Minimum hold period
        if bars_held < MIN_HOLD_BARS and position != 0:
            signals[i] = position * SIGNAL_SIZE
            continue
        
        # Wick rejection signal: small body with strong close
        # Long rejection: close near high, small body
        # Short rejection: close near low, small body
        close_to_high = (high[i] - close[i]) / total_range if total_range[i] > 0 else 0
        close_to_low = (close[i] - low[i]) / total_range if total_range[i] > 0 else 0
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Wick rejection conditions
        long_rejection = (body_ratio[i] < (1 - WICK_RATIO_THRESHOLD)) and \
                         (close_to_high < 0.2) and volume_ok  # Small body, close near high
        short_rejection = (body_ratio[i] < (1 - WICK_RATIO_THRESHOLD)) and \
                          (close_to_low < 0.2) and volume_ok   # Small body, close near low
        
        # Generate signals
        if position == 0:
            if long_rejection:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_rejection:
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