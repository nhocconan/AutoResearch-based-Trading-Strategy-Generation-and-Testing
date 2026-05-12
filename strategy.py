#!/usr/bin/env python3
# 1d_1W_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 1d to determine trend direction,
# combined with RSI for momentum confirmation and Choppiness Index for regime filtering.
# Enter long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2).
# Enter short when KAMA turns down, RSI < 50, and market is trending.
# Uses weekly timeframe for trend confirmation to avoid counter-trend trades.
# Designed for low frequency (10-25 trades/year) to minimize fee drag.

name = "1d_1W_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(np.diff(close, prepend=close[0])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.where((hh - ll) != 0, 
                    100 * np.log10(atr / (hh - ll)) / np.log10(period), 
                    50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate daily KAMA for trend direction
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_dir = np.diff(kama, prepend=0)  # positive = rising, negative = falling

    # Calculate daily RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Calculate daily Choppiness Index for regime filter
    chop = calculate_chop(high, low, close, period=14)
    trending = chop < 38.2  # Trending market
    ranging = chop > 61.8   # Ranging market

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter: only trade in direction of weekly trend
        bullish_weekly = close[i] > ema_1w_aligned[i]
        bearish_weekly = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: KAMA turning up, RSI > 50, trending market, and bullish weekly trend
            if (kama_dir[i] > 0 and rsi[i] > 50 and trending[i] and bullish_weekly):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down, RSI < 50, trending market, and bearish weekly trend
            elif (kama_dir[i] < 0 and rsi[i] < 50 and trending[i] and bearish_weekly):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down OR RSI < 40 OR market becomes ranging OR weekly trend turns bearish
            if (kama_dir[i] < 0 or rsi[i] < 40 or ranging[i] or not bullish_weekly):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up OR RSI > 60 OR market becomes ranging OR weekly trend turns bullish
            if (kama_dir[i] > 0 or rsi[i] > 60 or ranging[i] or not bearish_weekly):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals