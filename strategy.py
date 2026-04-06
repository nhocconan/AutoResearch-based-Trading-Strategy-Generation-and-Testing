#!/usr/bin/env python3
"""
exp_12451_6d_hma_cross_1d_rsi_regime_v1
Hypothesis: 6H Hull Moving Average crossover with 1D RSI regime filter
- Uses HMA(9/21) crossover for momentum signals on 6H
- Filters by 1D RSI(14) to avoid counter-trend trades: long only when RSI>50, short only when RSI<50
- Volume confirmation ensures momentum behind moves
- Designed to work in both bull (follows trend) and bear (avoids false breaks in wrong RSI regime)
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12451_6d_hma_cross_1d_rsi_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
HMA_FAST = 9
HMA_SLOW = 21
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def wma(values, window):
    """Weighted Moving Average"""
    weights = np.arange(1, window + 1)
    return np.convolve(values, weights, 'valid') / weights.sum()

def hma(series, period):
    """Hull Moving Average"""
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt)

def rsi(close, period):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(high, low, close, period):
    """Average True Range"""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI for regime filter
    rsi_1d = rsi(df_1d['close'].values, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6H indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # HMA indicators
    hma_fast = hma(close, HMA_FAST)
    hma_slow = hma(close, HMA_SLOW)
    
    # Volume and ATR
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_val = atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HMA_SLOW, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily RSI not available
        if np.isnan(rsi_1d_aligned[i]):
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
        
        # RSI regime filter (daily)
        rsi_long_regime = rsi_1d_aligned[i] > 50  # bullish regime
        rsi_short_regime = rsi_1d_aligned[i] < 50  # bearish regime
        
        # HMA crossover signals
        hma_cross_up = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_down = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # Entry conditions with regime filter
        long_entry = volume_ok and hma_cross_up and rsi_long_regime
        short_entry = volume_ok and hma_cross_down and rsi_short_regime
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_val[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_val[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals