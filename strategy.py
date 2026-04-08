# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_kama_rsi_chop_filter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for momentum and Choppiness Index for regime filtering.
Only take long when KAMA upward, RSI > 50, and market is trending (CHOP < 61.8).
Only take short when KAMA downward, RSI < 50, and market is trending (CHOP < 61.8).
This avoids whipsaw in ranging markets and captures strong trends.
Weekly trend filter ensures alignment with higher timeframe momentum.
Designed to work in both bull (trend following) and bear (avoids false signals in chop) markets.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def _kama(close, er_length=10, fast_ema=2, slow_ema=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else np.sum(np.abs(np.diff(close)))
    # Handle array case properly
    if len(change) > 1:
        volatility = pd.Series(change).rolling(window=er_length, min_periods=1).sum().values
    else:
        volatility = np.full_like(change, 1.0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def _rsi(close, length=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
    avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _choppiness_index(high, low, close, length=14):
    """Choppiness Index: high values indicate ranging, low values indicate trending"""
    atr = np.abs(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
    highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
    lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
    range_max_min = highest_high - lowest_low
    chop = np.where(range_max_min != 0, -100 * np.log10(atr_sum / range_max_min) / np.log10(length), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA for trend direction
    kama = _kama(close, er_length=10, fast_ema=2, slow_ema=30)
    
    # Calculate RSI for momentum
    rsi = _rsi(close, length=14)
    
    # Calculate Choppiness Index for regime filtering
    chop = _choppiness_index(high, low, close, length=14)
    
    # Weekly trend filter: use 1-week KAMA direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        kama_1w = _kama(df_1w['close'].values, er_length=10, fast_ema=2, slow_ema=30)
        kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    else:
        kama_1w_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine if market is trending (CHOP < 61.8) or ranging (CHOP >= 61.8)
        is_trending = chop[i] < 61.8
        
        # KAMA direction: comparing current to previous value
        kama_up = kama[i] > kama[i-1] if i > 0 else False
        kama_down = kama[i] < kama[i-1] if i > 0 else False
        
        # Weekly KAMA direction filter
        weekly_kama_up = kama_1w_aligned[i] > kama_1w_aligned[i-1] if i > 0 and not np.isnan(kama_1w_aligned[i-1]) else False
        weekly_kama_down = kama_1w_aligned[i] < kama_1w_aligned[i-1] if i > 0 and not np.isnan(kama_1w_aligned[i-1]) else False
        
        # Long conditions: KAMA up, RSI > 50, trending market, weekly alignment
        if (kama_up and rsi[i] > 50 and is_trending and 
            (np.isnan(kama_1w_aligned[i]) or weekly_kama_up)):
            signals[i] = 0.25
            
        # Short conditions: KAMA down, RSI < 50, trending market, weekly alignment
        elif (kama_down and rsi[i] < 50 and is_trending and 
              (np.isnan(kama_1w_aligned[i]) or weekly_kama_down)):
            signals[i] = -0.25
            
        # Otherwise, stay flat
        else:
            signals[i] = 0.0
    
    return signals