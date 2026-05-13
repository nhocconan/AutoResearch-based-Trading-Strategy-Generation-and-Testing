#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) to determine trend direction,
# combine with RSI for momentum confirmation and Choppiness Index to filter ranging markets.
# Enter long when KAMA trends up, RSI > 50, and market is trending (CHOP < 38.2).
# Enter short when KAMA trends down, RSI < 50, and market is trending (CHOP < 38.2).
# Uses 1-week timeframe for trend confirmation to avoid false signals in sideways markets.
# Designed for low turnover (~15-25 trades/year) to minimize fee impact and work in both bull and bear markets.

name = "1d_KAMA_Trend_RSI_Chop_Filter"
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

    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i-1] * (close_1d[i] - kama[i-1])

    # Calculate RSI on 1d close (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])

    # Calculate Choppiness Index on 1d (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    # Sum of TR over 14 periods
    atr_sum = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    hh = np.full_like(high, np.nan)
    ll = np.full_like(low, np.nan)
    for i in range(14, len(high)):
        hh[i] = np.max(high[i-13:i+1])
        ll[i] = np.min(low[i-13:i+1])
    # Chop calculation
    chop = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if atr_sum[i] > 0 and (hh[i] - ll[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = np.nan

    # Calculate 1w EMA20 for trend confirmation
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Align all 1d indicators to lower timeframe (they're already 1d, but we need to align to same index as prices)
    # Since prices is 1d, we can use directly but need to handle length mismatch
    # Create series indexed by df_1d index then reindex to prices index
    close_1d_series = pd.Series(close_1d, index=df_1d.index)
    kama_series = pd.Series(kama, index=df_1d.index)
    rsi_series = pd.Series(rsi, index=df_1d.index)
    chop_series = pd.Series(chop, index=df_1d.index)
    
    # Reindex to prices index (forward fill to handle missing dates)
    kama_aligned = kama_series.reindex(prices.index, method='ffill').values
    rsi_aligned = rsi_series.reindex(prices.index, method='ffill').values
    chop_aligned = chop_series.reindex(prices.index, method='ffill').values

    # Align 1w EMA to prices index
    close_1w_series = pd.Series(close_1w, index=df_1w.index)
    ema_1w_series = pd.Series(ema_1w, index=df_1w.index)
    ema_1w_aligned = ema_1w_series.reindex(prices.index, method='ffill').values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # start after warmup
        # Skip if data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA trending up, RSI > 50, trending market (CHOP < 38.2), price above weekly EMA
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 38.2 and 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down, RSI < 50, trending market (CHOP < 38.2), price below weekly EMA
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 38.2 and 
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below KAMA or market becomes ranging (CHOP > 61.8)
            if close[i] < kama_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above KAMA or market becomes ranging (CHOP > 61.8)
            if close[i] > kama_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals