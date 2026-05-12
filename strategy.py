#!/usr/bin/env python3
# 1d_KAMA_Trend_With_Weekly_Trend_Filter
# Hypothesis: KAMA adapts to market noise, providing a smooth trend line that reduces whipsaw. In trending markets (weekly trend aligned), we take KAMA crossovers with volume confirmation. In ranging markets (weekly trend weak), we avoid trades. Designed for 1d timeframe to capture major swings with low trade frequency, suitable for both bull and bear markets via trend-following logic.

name = "1d_KAMA_Trend_With_Weekly_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Weekly EMA20 trend filter (smoothed trend)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # KAMA (adaptive moving average) parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility properly: sum of absolute changes over er_len period
    volatility = np.zeros_like(change)
    for i in range(er_len, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_len:i])))

    # Avoid division by zero
    er = np.zeros_like(close)
    er[er_len:] = change / np.where(volatility[er_len:] == 0, 1, volatility[er_len:])
    er = np.where(er > 1, 1, er)  # cap at 1

    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # start with first valid close

    # Calculate KAMA
    for i in range(er_len + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(er_len + 1, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]

        if position == 0:
            # LONG: Price crosses above KAMA with volume and weekly uptrend
            if i > 0 and not np.isnan(kama[i-1]) and close[i-1] <= kama[i-1] and close[i] > kama[i] and volume_ok[i] and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with volume and weekly downtrend
            elif i > 0 and not np.isnan(kama[i-1]) and close[i-1] >= kama[i-1] and close[i] < kama[i] and volume_ok[i] and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or weekly trend turns down
            if i > 0 and not np.isnan(kama[i-1]) and close[i-1] >= kama[i-1] and close[i] < kama[i] or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or weekly trend turns up
            if i > 0 and not np.isnan(kama[i-1]) and close[i-1] <= kama[i-1] and close[i] > kama[i] or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals