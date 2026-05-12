#!/usr/bin/env python3
"""
1d_KAMA_WeeklyTrend_RangeFilter
Hypothesis: Use daily KAMA to identify trend and range conditions, with weekly trend filter
to avoid counter-trend trades. Combines adaptive trend following with range avoidance.
Works in bull by following trends, in bear by avoiding whipsaws in ranging markets.
Target: 15-25 trades/year.
"""

name = "1d_KAMA_WeeklyTrend_RangeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    wk_close = df_1w['close'].values
    wk_ema = pd.Series(wk_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA to daily
    wk_ema_aligned = align_htf_to_ltf(prices, df_1w, wk_ema)

    # Daily KAMA for trend direction
    daily_kama = calculate_kama(close, er_len=10, fast=2, slow=30)
    daily_kama_prev = np.roll(daily_kama, 1)
    daily_kama_prev[0] = daily_kama[0]

    # Daily ATR for volatility filter
    daily_atr = calculate_atr(high, low, close, period=14)
    atr_ma = pd.Series(daily_atr).rolling(window=20, min_periods=20).mean().values
    low_volatility = daily_atr < (0.8 * atr_ma)  # Range condition: low volatility

    # Volume confirmation: current volume > 1.2x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.2 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(wk_ema_aligned[i]) or np.isnan(daily_kama[i]) or 
            np.isnan(daily_kama_prev[i]) or np.isnan(low_volatility[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above KAMA, weekly uptrend, not in low volatility, volume ok
            if (close[i] > daily_kama[i] and 
                wk_ema_aligned[i] > wk_ema_aligned[max(0, i-1)] and  # Weekly EMA rising
                not low_volatility[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA, weekly downtrend, not in low volatility, volume ok
            elif (close[i] < daily_kama[i] and 
                  wk_ema_aligned[i] < wk_ema_aligned[max(0, i-1)] and  # Weekly EMA falling
                  not low_volatility[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below KAMA OR weekly trend turns down
            if (close[i] < daily_kama[i] or 
                wk_ema_aligned[i] < wk_ema_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above KAMA OR weekly trend turns up
            if (close[i] > daily_kama[i] or 
                wk_ema_aligned[i] > wk_ema_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals