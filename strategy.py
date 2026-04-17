#!/usr/bin/env python3
"""
6h_12h_1d_LongTermTrend_WithPullback
Hypothesis: On 6h timeframe, enter long when price is above 12h EMA50 and pulls back to 12h EMA21 with RSI < 40; enter short when price is below 12h EMA50 and pulls back to 12h EMA21 with RSI > 60. Uses 1d ADX > 25 to filter for trending markets only. Designed to capture trend continuations in both bull and bear markets with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate EMA with proper handling of NaN values."""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_ma / (tr_ma + 1e-10)
    minus_di = 100 * dm_minus_ma / (tr_ma + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 12h Data (HTF for trend and pullback) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMAs
    ema21_12h = calculate_ema(close_12h, 21)
    ema50_12h = calculate_ema(close_12h, 50)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h RSI
    rsi_12h = calculate_rsi(close_12h, 14)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # === 1d Data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_1d_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA50 and pulling back to EMA21 with RSI < 40
            if close[i] > ema50_12h_aligned[i] and low[i] <= ema21_12h_aligned[i] and rsi_12h_aligned[i] < 40:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA50 and pulling back to EMA21 with RSI > 60
            elif close[i] < ema50_12h_aligned[i] and high[i] >= ema21_12h_aligned[i] and rsi_12h_aligned[i] > 60:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below EMA21 or RSI > 60 (overbought)
            if close[i] < ema21_12h_aligned[i] or rsi_12h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above EMA21 or RSI < 40 (oversold)
            if close[i] > ema21_12h_aligned[i] or rsi_12h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_1d_LongTermTrend_WithPullback"
timeframe = "6h"
leverage = 1.0