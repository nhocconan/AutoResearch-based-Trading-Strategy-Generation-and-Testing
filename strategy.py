#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum,
# and Choppiness Index to filter range-bound markets. Only trade when KAMA slope aligns with RSI
# and market is trending (CHOP < 38.2). Designed for 1d timeframe to achieve 7-25 trades/year.
# Works in both bull and bear markets by adapting to trend strength and avoiding chop.

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(close) > 1 else 0
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initialize first average
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[1:period])
        avg_loss[period-1] = np.mean(loss[1:period])
    
    # Wilder smoothing
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr[period-1] = np.mean(tr[1:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Sum of ATR over period
    atr_sum = np.zeros_like(close)
    for i in range(period-1, len(close)):
        atr_sum[i] = np.sum(atr[i-period+1:i+1])
    
    # Max high - min low over period
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(period-1, len(close)):
        max_high[i] = np.max(high[i-period+1:i+1])
        min_low[i] = np.min(low[i-period+1:i+1])
    
    # Avoid division by zero
    range_hl = max_high - min_low
    chop = np.where(range_hl != 0, 100 * np.log10(atr_sum / range_hl) / np.log10(period), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for higher timeframe trend filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate KAMA on 1h close
    kama_1h = calculate_kama(close_1h, er_period=10, fast_ema=2, slow_ema=30)
    # Calculate slope of KAMA (1-period change)
    kama_slope_1h = np.diff(kama_1h, prepend=kama_1h[0])
    # Align KAMA slope to daily timeframe
    kama_slope_1h_aligned = align_htf_to_ltf(prices, df_1h, kama_slope_1h)
    
    # Calculate RSI on daily close
    rsi = calculate_rsi(close, period=14)
    
    # Calculate Choppiness Index on daily data
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_slope_1h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (bullish trend), RSI > 50 (bullish momentum), CHOP < 38.2 (trending market)
            if kama_slope_1h_aligned[i] > 0 and rsi[i] > 50 and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (bearish trend), RSI < 50 (bearish momentum), CHOP < 38.2 (trending market)
            elif kama_slope_1h_aligned[i] < 0 and rsi[i] < 50 and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA falling or RSI < 40 or CHOP > 61.8 (choppy market)
            if kama_slope_1h_aligned[i] < 0 or rsi[i] < 40 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA rising or RSI > 60 or CHOP > 61.8 (choppy market)
            if kama_slope_1h_aligned[i] > 0 or rsi[i] > 60 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals