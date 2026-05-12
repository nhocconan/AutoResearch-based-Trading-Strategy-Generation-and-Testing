#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_14_ChopFilter
Hypothesis: KAMA direction determines trend (long when rising, short when falling) on daily timeframe.
Entry confirmed by RSI(14) not in extreme overbought/oversold (>70 or <30) and choppy market filter (Chop > 61.8) to avoid trending whipsaws.
Exits when KAMA direction reverses or RSI reaches extreme. Works in both bull/bear by following adaptive trend.
"""

name = "1d_KAMA_Direction_RSI_14_ChopFilter"
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

    # Get weekly data ONCE before loop for Chop filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate KAMA on daily data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum |close(t) - close(t-1)| over 10 periods
    # Fix dimensions: change length is n-9, volatility length is n-1
    # We'll compute ER using pandas for simplicity and correct alignment
    close_1d_series = pd.Series(close_1d)
    diff = close_1d_series.diff(10)
    abs_diff = close_1d_series.diff(1).abs()
    er = np.abs(diff) / abs_diff.rolling(10, min_periods=10).sum()
    er = er.fillna(0).values  # ER = 0 when no volatility

    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])

    # Shift KAMA to avoid look-ahead (use previous day's KAMA)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]  # first value
    kama_direction = kama_prev > np.roll(kama_prev, 1)  # rising if today's KAMA > yesterday's
    kama_direction = np.roll(kama_direction, 1)  # shift to align with current bar
    kama_direction[0] = False  # no direction on first bar

    # Align KAMA direction to daily timeframe (already daily, but use for consistency)
    kama_direction_aligned = align_htf_to_ltf(prices, df_1d, kama_direction.astype(float))

    # RSI(14) on daily data
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    # Choppy market filter on weekly data: Chop > 61.8 = ranging (good for mean reversion in trend)
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift())
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).sum()
    # Chop = 100 * log15(sum(tr1,14) / (atr14 * 14)) / log15(14)
    sum_tr14 = tr.rolling(14, min_periods=14).sum()
    chop = 100 * (np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14))
    chop = chop.fillna(50).values  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after warmup
        if (np.isnan(kama_direction_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising AND RSI not overbought AND choppy market (range)
            if (kama_direction_aligned[i] == True and 
                rsi_aligned[i] < 70 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling AND RSI not oversold AND choppy market (range)
            elif (kama_direction_aligned[i] == False and 
                  rsi_aligned[i] > 30 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI overbought
            if (kama_direction_aligned[i] == False or 
                rsi_aligned[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI oversold
            if (kama_direction_aligned[i] == True or 
                rsi_aligned[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals