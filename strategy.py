#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: On daily timeframe, KAMA captures adaptive trend direction. RSI(14) filters extremes (avoid buying overbought/selling oversold). Choppiness Index (CHOP) regime filter: only trade when CHOP > 50 (range-bound conditions) to avoid whipsaws in strong trends. This combination should work in both bull and bear markets by adapting to volatility and avoiding false signals in strong trends. Targets 10-20 trades per year to minimize fee drag.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate KAMA (adaptive trend) on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14) for overbought/oversold filter
    def rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([np.array([np.nan]), delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.nansum(gain[1:period+1]) / period
        avg_loss[period] = np.nansum(loss[1:period+1]) / period
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_vals = rsi(close, 14)

    # Choppiness Index (CHOP) for regime filter
    def chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TRUE RANGE over period
        tr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            tr_sum[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period, len(close)):
            max_high[i] = np.nanmax(high[i-period+1:i+1])
            min_low[i] = np.nanmin(low[i-period+1:i+1])
        
        # CHOP formula
        chop = 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(period)
        # Handle division by zero or invalid cases
        chop = np.where((max_high - min_low) != 0, chop, 50)
        return chop

    chop_vals = chop(high, low, close, 14)

    # Weekly EMA34 for trend filter (only trade in direction of weekly trend)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA (uptrend) + RSI < 60 (not overbought) + CHOP > 50 (range) + weekly uptrend
            if (close[i] > kama[i] and 
                rsi_vals[i] < 60 and 
                chop_vals[i] > 50 and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA (downtrend) + RSI > 40 (not oversold) + CHOP > 50 (range) + weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi_vals[i] > 40 and 
                  chop_vals[i] > 50 and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR RSI > 70 (overbought) OR CHOP < 30 (strong trend)
            if close[i] < kama[i] or rsi_vals[i] > 70 or chop_vals[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR RSI < 30 (oversold) OR CHOP < 30 (strong trend)
            if close[i] > kama[i] or rsi_vals[i] < 30 or chop_vals[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals