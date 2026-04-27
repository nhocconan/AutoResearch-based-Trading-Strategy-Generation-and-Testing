#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Long when KAMA rising and RSI < 50 in choppy markets (CHOP > 61.8)
Short when KAMA falling and RSI > 50 in choppy markets (CHOP > 61.8)
Exit when RSI crosses 50 or chop drops below 38.2 (trending)
Designed for mean-reversion in chop, trend-following in trending markets.
Uses 1w trend filter to avoid counter-trend trades in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.empty(n, dtype=np.float64)
    kama[:] = np.nan
    
    if n < er_length:
        return kama
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))  # |close[i] - close[i-er_length]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over er_length
    
    # Pad arrays
    change = np.concatenate([np.full(er_length-1, np.nan), change])
    volatility = np.concatenate([np.full(er_length-1, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama[er_length-1] = close[er_length-1]  # Start with close
    for i in range(er_length, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_chop(high, low, close, period=14):
    """Choppiness Index"""
    n = len(close)
    chop = np.empty(n, dtype=np.float64)
    chop[:] = np.nan
    
    if n < period:
        return chop
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of TR over period
    atr_sum = np.nancumsum(tr)
    atr_sum = np.concatenate([np.full(period, np.nan), atr_sum[period:] - atr_sum[:-period]])
    
    # Highest high and lowest low over period
    highest_high = np.concatenate([np.full(period-1, np.nan), 
                                   [np.max(high[i-period+1:i+1]) if i >= period-1 else np.nan 
                                    for i in range(period-1, n)]])
    lowest_low = np.concatenate([np.full(period-1, np.nan), 
                                 [np.min(low[i-period+1:i+1]) if i >= period-1 else np.nan 
                                  for i in range(period-1, n)]])
    
    # Chop calculation
    # Avoid log10(0) and division by zero
    range_hl = highest_high - lowest_low
    chop = np.where((range_hl != 0) & (atr_sum != 0), 
                    100 * np.log10(atr_sum / range_hl) / np.log10(period), 
                    50)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA, RSI, Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate indicators on 1d
    kama = calculate_kama(df_1d['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    rsi = np.empty_like(df_1d['close'].values)
    rsi[:] = np.nan
    
    # RSI(14)
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.nancumsum(gain)
    avg_loss = np.nancumsum(loss)
    avg_gain = np.concatenate([np.full(14, np.nan), avg_gain[14:] - avg_gain[:-14]])
    avg_loss = np.concatenate([np.full(14, np.nan), avg_loss[14:] - avg_loss[:-14]])
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan
    
    chop = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # 1w EMA20 for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    
    # Align to 1d timeframe (our trading timeframe)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Warmup
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        kama_val = kama_aligned[i]
        kama_prev = kama_aligned[i-1] if i > 0 else kama_val
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        close_val = close[i]
        
        # KAMA direction
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        # Chop regimes
        chop_choppy = chop_val > 61.8  # Range/market
        chop_trending = chop_val < 38.2  # Trending market
        
        if position == 0:
            # Enter long: KAMA rising + RSI < 50 in choppy market
            # OR in trending market, follow 1w trend (only long if price > 1w EMA)
            if chop_choppy:
                if kama_rising and rsi_val < 50:
                    signals[i] = size
                    position = 1
            else:  # Trending or neutral chop
                # Only take trend-aligned trades
                if close_val > ema_1w_val and kama_rising and rsi_val < 50:
                    signals[i] = size
                    position = 1
            # Enter short: KAMA falling + RSI > 50 in choppy market
            # OR in trending market, follow 1w trend (only short if price < 1w EMA)
            if chop_choppy:
                if kama_falling and rsi_val > 50:
                    signals[i] = -size
                    position = -1
            else:  # Trending or neutral chop
                if close_val < ema_1w_val and kama_falling and rsi_val > 50:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 OR chop drops (trending starts) AND RSI > 50
            if rsi_val > 50 or (chop_val < 38.2 and rsi_val > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses below 50 OR chop drops (trending starts) AND RSI < 50
            if rsi_val < 50 or (chop_val < 38.2 and rsi_val < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0