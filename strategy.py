#!/usr/bin/env python3
"""
1d_KAMA_Direction_With_RSI_and_Choppiness_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
Combined with RSI for momentum confirmation and Choppiness Index to filter ranging conditions (avoid false signals).
Designed for 1d timeframe to capture major trends with minimal trades (7-25/year), reducing fee impact.
Works in bull markets via trend following and in bear markets via avoidance of false signals during ranging periods.
"""

name = "1d_KAMA_Direction_With_RSI_and_Choppiness_Filter"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate KAMA (adaptive moving average) on daily close
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[9:], np.roll(close, 1)[9:]))  # |close - close[10]|
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1]))).reshape(-1, 1)  # placeholder for rolling sum
    # Proper rolling volatility calculation
    volatility = np.abs(np.subtract(close[1:], close[:-1]))
    volatility = np.concatenate([[np.nan], volatility])  # align with close index
    er = np.full_like(close, np.nan)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            vol_sum = np.sum(np.abs(np.subtract(close[i-9:i+1], close[i-10:i])))  # volatility over 10 periods
            if vol_sum > 0:
                er[i] = price_change / vol_sum
            else:
                er[i] = 0
    # Smooth ER with smoothing constants
    sc = (er * 0.59 + 0.06) ** 2  # where 0.59 = 2/(2+1), 0.06 = 2/(30+1) - 2/(2+1)
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Align KAMA to daily timeframe (already aligned as we used daily close)
    kama_aligned = kama  # no alignment needed as calculated on same timeframe

    # Calculate RSI(14) on daily close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Calculate Choppiness Index(14) on daily data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    # Sum of TR over 14 periods
    tr_sum = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    # Chop calculation
    chop = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral if undefined

    # Get weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema34_val = ema34_1w_aligned[i]

        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(ema34_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI > 50 (bullish momentum) + Chop < 61.8 (trending market)
            if close[i] > kama_val and rsi_val > 50 and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI < 50 (bearish momentum) + Chop < 61.8 (trending market)
            elif close[i] < kama_val and rsi_val < 50 and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or RSI < 40 (momentum loss) or Chop > 61.8 (ranging market)
            if close[i] < kama_val or rsi_val < 40 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or RSI > 60 (momentum loss) or Chop > 61.8 (ranging market)
            if close[i] > kama_val or rsi_val > 60 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals