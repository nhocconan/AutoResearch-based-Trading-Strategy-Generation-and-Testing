#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend with daily volume confirmation and weekly ATR regime filter.
# KAMA adapts to market noise, reducing false signals in choppy markets. Volume confirms
# institutional participation. Weekly ATR regime filter avoids trading in extreme volatility.
# Works in bull markets (trend following) and bear markets (mean reversion at extremes).

name = "exp_13352_12h_kama_volume_atr_regime_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
KAMA_FAST = 2
KAMA_SLOW = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
ATR_MA_PERIOD = 50
SIGNAL_SIZE = 0.25

def calculate_kama(close, fast, slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, slow))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle first element
    volatility = np.insert(volatility, 0, 0)
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly ATR for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, ATR_PERIOD)
    atr_ma_1w = pd.Series(atr_1w).rolling(window=ATR_MA_PERIOD, min_periods=ATR_MA_PERIOD).mean().values
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA
    kama = calculate_kama(close, KAMA_FAST, KAMA_SLOW)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_SLOW, VOLUME_MA_PERIOD, ATR_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(kama[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_ma_1w_aligned[i]) or np.isnan(atr[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime filter: avoid extreme volatility (weekly ATR > 1.5 * MA)
        volatility_regime = atr_1w[i] < (1.5 * atr_ma_1w_aligned[i])
        
        # Trend signal: price vs KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # Generate signals
        if position == 0:
            if volume_ok and volatility_regime and above_kama:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif volume_ok and volatility_regime and below_kama:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals