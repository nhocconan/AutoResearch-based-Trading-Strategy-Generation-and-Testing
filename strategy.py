#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13935_6d_camarilla1d_fade_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h Camarilla pivot fade strategy using daily pivots
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend follow)
# Works in both bull and bear by adapting to price action at key levels
# Target: 50-150 total trades over 4 years by using strict pivot levels
# Volume confirmation reduces false signals

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    pivot = (high + low + close) / 3
    r4 = pivot + (range_val * 1.1)
    r3 = pivot + (range_val * 0.55)
    s3 = pivot - (range_val * 0.55)
    s4 = pivot - (range_val * 1.1)
    return r4, r3, s3, s4

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    r4_1d_a = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_a = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_a = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_a = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for entry
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for overbought/oversold confirmation
    rsi = calculate_rsi(close, 14)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_a[i]) or np.isnan(r3_1d_a[i]) or np.isnan(s3_1d_a[i]) or np.isnan(s4_1d_a[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Fade logic: sell at R3, buy at S3 with RSI extremes
        fade_short = volume_ok and (close[i] >= r3_1d_a[i]) and (rsi[i] > 60)
        fade_long = volume_ok and (close[i] <= s3_1d_a[i]) and (rsi[i] < 40)
        
        # Breakout logic: buy at R4 break, sell at S4 break
        breakout_long = volume_ok and (close[i] > r4_1d_a[i])
        breakout_short = volume_ok and (close[i] < s4_1d_a[i])
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif fade_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.5 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade at R3 or stop loss
            if close[i] >= r3_1d_a[i] or close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade at S3 or stop loss
            if close[i] <= s3_1d_a[i] or close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals