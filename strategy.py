#!/usr/bin/env python3
"""
12h_1d_KAMA_RSI_Chop_Filter
Hypothesis: Uses KAMA direction from daily timeframe for trend, RSI for momentum, and Choppiness Index for regime filtering.
Enters long when daily KAMA is up, RSI < 30 (oversold), and market is choppy (CHOP > 61.8). Enters short when daily KAMA is down,
RSI > 70 (overbought), and market is choppy. Uses 12h timeframe for execution with tight entry conditions to limit trades to 12-37/year.
Designed to work in both bull and bear markets by fading extremes in choppy regimes while following higher-timeframe trend.
"""

name = "12h_1d_KAMA_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i < er_period:
            er[i] = 0
        else:
            volatility_sum = np.sum(volatility[i-er_period+1:i+1])
            if volatility_sum > 0:
                er[i] = np.abs(close[i] - close[i-er_period]) / volatility_sum
            else:
                er[i] = 0
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Daily KAMA for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast=2, slow=30)
    kama_1d_dir = np.where(kama_1d > np.roll(kama_1d, 1), 1, -1)
    kama_1d_dir[0] = 1
    
    # Align daily KAMA direction to 12h timeframe
    kama_1d_dir_12h = align_htf_to_ltf(prices, df_1d, kama_1d_dir)
    
    # --- 12h RSI ---
    rsi = calculate_rsi(close, period=14)
    
    # --- 12h Choppiness Index ---
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_dir_12h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: choppy market (CHOP > 61.8)
        is_choppy = chop[i] > 61.8
        
        if position == 0:
            # Long: daily KAMA up + RSI oversold + choppy market
            if (kama_1d_dir_12h[i] == 1 and 
                rsi[i] < 30 and 
                is_choppy):
                signals[i] = 0.25
                position = 1
            # Short: daily KAMA down + RSI overbought + choppy market
            elif (kama_1d_dir_12h[i] == -1 and 
                  rsi[i] > 70 and 
                  is_choppy):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral or trend changes
            if position == 1:
                # Exit long: RSI > 50 or daily KAMA turns down
                if rsi[i] > 50 or kama_1d_dir_12h[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 or daily KAMA turns up
                if rsi[i] < 50 or kama_1d_dir_12h[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals