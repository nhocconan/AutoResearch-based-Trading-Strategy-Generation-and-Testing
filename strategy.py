#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12547_6d_camarilla1d_v4"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 10
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_EXIT_MULTIPLIER = 1.5  # for trailing exit

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

def calculate_camarilla(high, low, close, period):
    """Calculate Camarilla levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    s4 = close - (range_val * 1.1 / 2)
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    return s1, s2, s3, s4, r1, r2, r3, r4, pivot

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
              pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
               pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    s1, s2, s3, s4, r1, r2, r3, r4, pivot = calculate_camarilla(high_1d, low_1d, close_1d, CAMARILLA_PERIOD)
    
    # Align Camarilla levels to 6h timeframe
    s1_a = align_htf_to_ltf(prices, df_1d, s1)
    s2_a = align_htf_to_ltf(prices, df_1d, s2)
    s3_a = align_htf_to_ltf(prices, df_1d, s3)
    s4_a = align_htf_to_ltf(prices, df_1d, s4)
    r1_a = align_htf_to_ltf(prices, df_1d, r1)
    r2_a = align_htf_to_ltf(prices, df_1d, r2)
    r3_a = align_htf_to_ltf(prices, df_1d, r3)
    r4_a = align_htf_to_ltf(prices, df_1d, r4)
    pivot_a = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 10
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(s1_a[i]) or np.isnan(r1_a[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss and trailing exit
        if position == 1:  # long position
            if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
            # Trail stop: exit if price drops ATR_EXIT_MULTIPLIER from high
            highest_since_entry = max(highest_since_entry, high[i])
            if close[i] <= highest_since_entry - (ATR_EXIT_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
            # Trail stop: exit if price rises ATR_EXIT_MULTIPLIER from low
            lowest_since_entry = min(lowest_since_entry, low[i])
            if close[i] >= lowest_since_entry + (ATR_EXIT_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # ADX filter for trending markets
        trending = adx[i] > 25 if not np.isnan(adx[i]) else False
        
        # Camarilla breakout/breakdown conditions
        long_breakout = close[i] > r3_a[i] and trending  # break above R3 in trend
        short_breakdown = close[i] < s3_a[i] and trending  # break below S3 in trend
        long_mean_revert = close[i] < s3_a[i] and not trending  # mean revert at S3 in range
        short_mean_revert = close[i] > r3_a[i] and not trending  # mean revert at R3 in range
        
        # Entry conditions
        long_entry = volume_ok and (long_breakout or long_mean_revert)
        short_entry = volume_ok and (short_breakdown or short_mean_revert)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals