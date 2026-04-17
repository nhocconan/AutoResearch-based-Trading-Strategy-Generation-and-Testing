#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Regime_V1
Strategy: 1-day KAMA direction + RSI(14) extremes + Choppiness Index regime filter.
Long: KAMA rising + RSI < 30 + CHOP > 61.8 (range)
Short: KAMA falling + RSI > 70 + CHOP > 61.8 (range)
Exit: Opposite signal or regime shift (CHOP < 38.2)
Position size: 0.25
Designed for mean reversion in ranging markets, avoids trending whipsaw.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(close[er_period:] - close[:-er_period])
    volatility = np.sum(np.abs(np.diff(close.reshape(-1, 1), axis=0)), axis=0)
    volatility = np.concatenate([np.full(er_period, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.full(er_period, np.nan), er])
    
    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    sc = np.concatenate([np.full(er_period, np.nan), sc])
    
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP)
    chop_period = 14
    atr_period = chop_period
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full_like(close, np.nan)
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period + 1, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of ATR over chop_period
    sum_atr = np.full_like(close, np.nan)
    for i in range(chop_period, n):
        sum_atr[i] = np.sum(atr[i-chop_period+1:i+1])
    
    # Max high - min low over chop_period
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(chop_period-1, n):
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    # Chop calculation
    chop = np.full_like(close, np.nan)
    for i in range(chop_period-1, n):
        if sum_atr[i] != 0 and max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
    
    # Weekly trend filter (1-week close > open = uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    trend_1w = (df_1w['close'] > df_1w['open']).astype(float).values  # 1 for up, 0 for down
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(er_period*2, rsi_period*2, chop_period) + 5, n):  # warmup
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Choppiness Index regime (range-bound)
        chop_range = chop[i] > 61.8  # chop > 61.8 = ranging market
        chop_trend = chop[i] < 38.2  # chop < 38.2 = trending market
        
        # Entry signals
        if position == 0:
            # Long: KAMA rising + RSI oversold + ranging market
            if kama_rising and rsi_oversold and chop_range:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI overbought + ranging market
            elif kama_falling and rsi_overbought and chop_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling OR RSI overbought OR trend shift
            if kama_falling or rsi[i] > 70 or chop_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI oversold OR trend shift
            if kama_rising or rsi[i] < 30 or chop_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime_V1"
timeframe = "1d"
leverage = 1.0