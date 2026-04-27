#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter Strategy
KAMA detects trend direction, RSI(14) filters overbought/oversold, Choppiness Index (14) filters regime.
Long when KAMA up, RSI < 50, Chop > 61.8 (range). Short when KAMA down, RSI > 50, Chop > 61.8.
Exit when KAMA reverses or Chop < 38.2 (trend).
Designed for 1d timeframe with 1w HTF trend filter. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_len=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.empty(n, dtype=np.float64)
    kama[:] = np.nan
    
    if n < er_len:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(n)
    er[er_len:] = change[er_len-1:] / volatility[er_len-1:]
    er[er_len-1] = 0  # first valid ER
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[er_len-1] = close[er_len-1]
    
    for i in range(er_len, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.empty(n, dtype=np.float64)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def choppiness_index(high, low, close, period=14):
    """Choppiness Index: 0 = trending, 100 = ranging"""
    n = len(close)
    chop = np.empty(n, dtype=np.float64)
    chop[:] = np.nan
    
    if n < period:
        return chop
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR
    atr = np.empty(n, dtype=np.float64)
    atr[:] = np.nan
    atr[period-1] = np.nanmean(tr[1:period+1])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Sum of ATR over period
    atr_sum = np.empty(n, dtype=np.float64)
    atr_sum[:] = np.nan
    for i in range(period-1, n):
        atr_sum[i] = np.nansum(atr[i-period+1:i+1])
    
    # Max-min over period
    max_high = np.empty(n, dtype=np.float64)
    min_low = np.empty(n, dtype=np.float64)
    max_high[:] = np.nan
    min_low[:] = np.nan
    for i in range(period-1, n):
        max_high[i] = np.max(high[i-period+1:i+1])
        min_low[i] = np.min(low[i-period+1:i+1])
    
    # Choppiness
    ratio = np.divide(atr_sum, max_high - min_low, out=np.full_like(atr_sum, np.nan), where=(max_high - min_low)!=0)
    chop = 100 * np.log10(ratio) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA on close
    kama_val = kama(df_1d['close'].values, er_len=10, fast=2, slow=30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_val)
    
    # RSI on close
    rsi_val = rsi(df_1d['close'].values, 14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_val)
    
    # Choppiness Index
    chop_val = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_val)
    
    # 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Warmup: need KAMA(30) + RSI(14) + Chop(14) + EMA50
    start_idx = max(30, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        kama_now = kama_aligned[i]
        rsi_now = rsi_aligned[i]
        chop_now = chop_aligned[i]
        ema_50_now = ema_50_aligned[i]
        
        # Trend filter: price vs weekly EMA50
        uptrend = price_now > ema_50_now
        downtrend = price_now < ema_50_now
        
        if position == 0:
            # Long: KAMA up, RSI < 50, Chop > 61.8 (range), uptrend
            if (kama_now > kama_aligned[i-1] and 
                rsi_now < 50 and 
                chop_now > 61.8 and 
                uptrend):
                signals[i] = size
                position = 1
            # Short: KAMA down, RSI > 50, Chop > 61.8 (range), downtrend
            elif (kama_now < kama_aligned[i-1] and 
                  rsi_now > 50 and 
                  chop_now > 61.8 and 
                  downtrend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA reverses down OR Chop < 38.2 (trend)
            if (kama_now < kama_aligned[i-1] or chop_now < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA reverses up OR Chop < 38.2 (trend)
            if (kama_now > kama_aligned[i-1] or chop_now < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0