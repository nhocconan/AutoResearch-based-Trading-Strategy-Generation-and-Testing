#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, 
combined with RSI for momentum confirmation and Choppiness Index for regime filtering. 
Enter long when KAMA slopes up, RSI > 50, and market is trending (CHOP < 38.2). 
Enter short when KAMA slopes down, RSI < 50, and market is trending (CHOP < 38.2). 
Exit when trend changes or market becomes choppy (CHOP > 61.8). 
Designed for low trade frequency (<25/year) to avoid fee decay while capturing sustained trends.
"""

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

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

    # Get 1-week data for trend filter (optional but adds robustness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, k=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Fix: vol needs to be rolling sum of absolute changes
    vol = np.zeros_like(close)
    for i in range(1, n):
        vol[i] = np.abs(close[i] - close[i-1])
    vol_sum = pd.Series(vol).rolling(window=10, min_periods=10).sum().values
    change_abs = np.abs(np.diff(close, k=10))
    # Pad change_abs to match length
    change_abs_padded = np.concatenate([np.full(10, np.nan), change_abs])
    er = np.where(vol_sum != 0, change_abs_padded / vol_sum, 0)
    # Smooth ER
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad rsi to align with close (first 14 values are NaN)
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi])

    # Choppiness Index (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    # Chop = 100 * log10(atr_sum / range_max_min) / log10(14)
    chop = np.where(range_max_min != 0, 100 * np.log10(atr_sum / range_max_min) / np.log10(14), 50)
    # Pad chop to align
    chop_padded = np.concatenate([np.full(13, np.nan), chop])  # 14-period needs 13 padding

    # Align 1-week close for trend filter
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_padded[i]) or 
            np.isnan(chop_padded[i]) or np.isnan(close_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # KAMA slope (direction)
        kama_slope = kama[i] - kama[i-1]

        if position == 0:
            # LONG: KAMA up, RSI > 50, trending market (CHOP < 38.2), price above 1w close
            if (kama_slope > 0 and 
                rsi_padded[i] > 50 and 
                chop_padded[i] < 38.2 and 
                close[i] > close_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, trending market (CHOP < 38.2), price below 1w close
            elif (kama_slope < 0 and 
                  rsi_padded[i] < 50 and 
                  chop_padded[i] < 38.2 and 
                  close[i] < close_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR market becomes choppy (CHOP > 61.8) OR trend breaks
            if (kama_slope < 0 or 
                chop_padded[i] > 61.8 or 
                close[i] < close_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR market becomes choppy (CHOP > 61.8) OR trend breaks
            if (kama_slope > 0 or 
                chop_padded[i] > 61.8 or 
                close[i] > close_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals