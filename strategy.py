#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h KAMA + RSI + Chop Filter (Daily Trend)
# Hypothesis: KAMA adapts to market noise, filtering out whipsaws. RSI confirms momentum,
# while daily trend filter ensures alignment with higher timeframe direction.
# Chop filter avoids ranging markets. Works in bull via KAMA crossovers with RSI > 50,
# in bear via shorts with RSI < 50, and avoids false signals in chop.
# Target: 20-30 trades/year to minimize fee drag.
name = "4h_kama_rsi_chop_daily_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close_10|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix dimensions: volatility needs same length as change
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align with change
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50
    
    # Chop Index (choppiness) - using 14-period
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    chop[:13] = 50  # insufficient data
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(daily_ema_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or daily trend turns bearish
            if close[i] < kama[i] or close[i] < daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or daily trend turns bullish
            if close[i] > kama[i] or close[i] > daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Avoid choppy markets (chop > 61.8 = ranging)
            if chop[i] > 61.8:
                signals[i] = 0.0
                continue
            # Enter long: price crosses above KAMA with RSI > 50 and bullish daily trend
            if close[i] > kama[i] and rsi[i] > 50 and close[i] > daily_ema_4h[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below KAMA with RSI < 50 and bearish daily trend
            elif close[i] < kama[i] and rsi[i] < 50 and close[i] < daily_ema_4h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals