#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: On 1d timeframe, KAMA direction (trend) + RSI extremes (mean reversion) 
combined with choppy regime filter (Chop > 61.8 = range, Chop < 38.2 = trend) 
provides edge in both bull and bear markets. Uses 1w ADX to confirm trend strength.
Targets 7-25 trades/year (30-100 total over 4 years) with low turnover.
Works in bull via KAMA trend + RSI pullbacks, bear via mean reversion at RSI extremes 
in choppy regimes with trend filter avoiding whipsaws.
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

    # Get 1w data for ADX trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate KAMA on daily close
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close - close[1]|
    # Handle volatility calculation properly for rolling sum
    volatility_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1, None], axis=1).flatten()))
    # Simpler approach: use pandas rolling sum
    volatility_series = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=10).sum()
    volatility = volatility_series.values
    volatility[:9] = np.nan  # first 9 values invalid
    er = np.zeros_like(close)
    er[:9] = np.nan
    er[10:] = change[10:] / volatility[10:]
    er = np.where(volatility != 0, er, 0)
    # Smoothing constants
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama = np.where(np.isnan(kama), close, kama)  # fill leading NaNs
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)  # KAMA is daily, align to 1d

    # Calculate RSI(14) on daily close
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # handle no loss case
    rsi = np.where(avg_gain == 0, 0, rsi)    # handle no gain case
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)  # RSI is daily

    # Calculate Choppiness Index on 1w data
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = np.abs(np.diff(high_1w, n=1))
    tr2 = np.abs(np.diff(low_1w, n=1))
    tr3 = np.abs(high_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_raw = np.where((max_high - min_low) == 0, 50, chop_raw)  # avoid div by zero
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)

    # Calculate 1w ADX for trend strength
    # +DM, -DM, TR
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr = np.maximum(np.abs(np.diff(high_1w, n=1)), 
                    np.maximum(np.abs(np.diff(low_1w, n=1)), 
                    np.abs(high_1w[1:] - close_1w[:-1])))
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(np.concatenate([[np.nan], dm_plus])).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(np.concatenate([[np.nan], dm_minus])).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    di_plus = 100 * dm_plus_smooth / atr_14
    di_minus = 100 * dm_minus_smooth / atr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Get aligned values for current 1d bar
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        adx_val = adx_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or 
            np.isnan(chop_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: Chop > 61.8 = range (mean revert), Chop < 38.2 = trend (trend follow)
        is_range = chop_val > 61.8
        is_trend = chop_val < 38.2

        if position == 0:
            # LONG conditions:
            # In trend: price > KAMA (uptrend) AND RSI < 40 (pullback)
            # In range: RSI < 30 (oversold) AND ADX > 20 (avoid dead market)
            if (is_trend and close[i] > kama_val and rsi_val < 40) or \
               (is_range and rsi_val < 30 and adx_val > 20):
                signals[i] = 0.25
                position = 1
            # SHORT conditions:
            # In trend: price < KAMA (downtrend) AND RSI > 60 (pullback)
            # In range: RSI > 70 (overbought) AND ADX > 20 (avoid dead market)
            elif (is_trend and close[i] < kama_val and rsi_val > 60) or \
                 (is_range and rsi_val > 70 and adx_val > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 
            # Trend: price < KAMA OR RSI > 70 (overbought)
            # Range: RSI > 70 OR chop > 70 (breaking out of range)
            if (is_trend and (close[i] < kama_val or rsi_val > 70)) or \
               (is_range and (rsi_val > 70 or chop_val > 70)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT:
            # Trend: price > KAMA OR RSI < 30 (oversold)
            # Range: RSI < 30 OR chop > 70 (breaking out of range)
            if (is_trend and (close[i] > kama_val or rsi_val < 30)) or \
               (is_range and (rsi_val < 30 or chop_val > 70)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals