#!/usr/bin/env python3
# 1D_KAMA_RSI_CHOP: KAMA direction + RSI(14) + Choppiness regime filter
# Long when KAMA rising, RSI > 55, CHOP > 61.8 (range) for mean reversion
# Short when KAMA falling, RSI < 45, CHOP > 61.8
# Exit when RSI crosses 50 or CHOP < 38.2 (trend)
# Uses 1w trend filter to avoid counter-trend trades in strong trends
# Target: 15-25 trades/year on 1d to minimize fee drag

name = "1D_KAMA_RSI_CHOP"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_len] = close[er_len]
        for i in range(er_len+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    # Calculate Choppiness Index
    def choppiness_index(high, low, close, cp_len=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        atr_sum = np.zeros_like(close)
        for i in range(cp_len, len(close)):
            atr_sum[i] = np.sum(atr[i-cp_len+1:i+1])
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(cp_len-1, len(close)):
            highest_high[i] = np.max(high[i-cp_len+1:i+1])
            lowest_low[i] = np.min(low[i-cp_len+1:i+1])
        chop = np.full_like(close, 50.0, dtype=float)
        for i in range(cp_len-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(cp_len)
        return chop

    kama_vals = kama(close, 10, 2, 30)
    chop = choppiness_index(high, low, close, 14)

    # RSI
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:length] = 50
        return rsi

    rsi_vals = rsi(close, 14)

    # 1w EMA40 for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)

    signals = np.zeros(n)
    position = 0

    for i in range(30, n):
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop[i]) or np.isnan(ema40_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        kama_rising = kama_vals[i] > kama_vals[i-1]
        kama_falling = kama_vals[i] < kama_vals[i-1]

        if position == 0:
            # LONG: KAMA rising, RSI > 55, CHOP > 61.8 (range), above 1w EMA40
            if (kama_rising and 
                rsi_vals[i] > 55 and 
                chop[i] > 61.8 and
                close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 45, CHOP > 61.8 (range), below 1w EMA40
            elif (kama_falling and 
                  rsi_vals[i] < 45 and 
                  chop[i] > 61.8 and
                  close[i] < ema40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 or CHOP < 38.2 (trend) or KAMA turns down
            if (rsi_vals[i] < 50 or 
                chop[i] < 38.2 or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 or CHOP < 38.2 (trend) or KAMA turns up
            if (rsi_vals[i] > 50 or 
                chop[i] < 38.2 or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals