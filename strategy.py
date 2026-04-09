#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: 1d Kaufman Adaptive Moving Average (KAMA) trend filter with 14-period RSI mean reversion and Choppiness Index regime filter.
# Uses 1d timeframe to minimize trade frequency (target: 7-25 trades/year). KAMA adapts to market noise, reducing false signals in ranging markets.
# RSI provides mean-reversion entries when extreme but aligned with KAMA trend direction. Choppiness Index (CHOP > 61.8) confirms ranging regime for mean reversion.
# Works in bull/bear markets: In trends (CHOP < 38.2), we avoid mean reversion trades; in ranges (CHOP > 61.8), we take RSI extremes in trend direction.
# Uses 1w HTF for major trend bias: only take longs when price > 1w EMA(50), shorts when price < 1w EMA(50).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    if len(close) < er_period:
        return np.full_like(close, np.nan, dtype=float)
    
    close = np.asarray(close)
    direction = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    close = np.asarray(close)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    
    atr = np.zeros_like(high)
    atr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        atr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr_sum = np.zeros_like(high)
    for i in range(period, len(high)):
        atr_sum[i] = np.sum(atr[i-period+1:i+1])
    
    hh = np.zeros_like(high)
    ll = np.zeros_like(high)
    for i in range(period-1, len(high)):
        hh[i] = np.max(high[i-period+1:i+1])
        ll[i] = np.min(low[i-period+1:i+1])
    
    chop = np.zeros_like(high)
    for i in range(period-1, len(high)):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        else:
            chop[i] = 50.0
    
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d KAMA(10,2,30) for trend
    kama_1d = calculate_kama(close_1d, er_period=10, fast_sc=2, slow_sc=30)
    
    # Align 1d KAMA to 1d timeframe (completed 1d candle only)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI(14)
    rsi_1d = calculate_rsi(close_1d, period=14)
    
    # Align 1d RSI to 1d timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Choppiness Index(14)
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, period=14)
    
    # Align 1d Chop to 1d timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1w HTF data for major trend bias (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(50)
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA OR RSI > 70 (overbought exit)
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA OR RSI < 30 (oversold exit)
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price > KAMA, RSI < 30 (oversold), CHOP > 61.8 (ranging), price > 1w EMA50 (bullish bias)
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                chop_1d_aligned[i] > 61.8 and 
                close[i] > ema_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price < KAMA, RSI > 70 (overbought), CHOP > 61.8 (ranging), price < 1w EMA50 (bearish bias)
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  chop_1d_aligned[i] > 61.8 and 
                  close[i] < ema_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals