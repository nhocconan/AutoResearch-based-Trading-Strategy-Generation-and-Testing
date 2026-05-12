# 1d_WeeklyTrend_KAMA_Reversal
# Hypothesis: On daily timeframe, KAMA (Kaufman Adaptive Moving Average) with ER=10 identifies adaptive trend direction.
# Long when price > KAMA and weekly trend bullish (weekly EMA34 > weekly EMA89), short when price < KAMA and weekly trend bearish.
# Weekly trend filter reduces whipsaw in choppy markets. Position size 0.25 for balanced risk/return.
# Designed to work in both bull (trend following) and bear (counter-trend reversals at extremes) via adaptive KAMA.
# Target: 10-25 trades/year to minimize fee drag.

name = "1d_WeeklyTrend_KAMA_Reversal"
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

    # Get weekly data for trend filter (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)

    # Calculate KAMA on daily close
    close_series = pd.Series(close)
    # Efficiency Ratio: |change over 10 periods| / sum of absolute changes over 10 periods
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)  # Handle division by zero
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate weekly EMA34 and EMA89 for trend filter
    close_weekly = pd.Series(df_weekly['close'].values)
    ema34_weekly = close_weekly.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_weekly = close_weekly.ewm(span=89, adjust=False, min_periods=89).mean().values

    # Align weekly EMAs to daily timeframe
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    ema89_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema89_weekly)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after KAMA warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(ema89_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        weekly_bullish = ema34_weekly_aligned[i] > ema89_weekly_aligned[i]
        weekly_bearish = ema34_weekly_aligned[i] < ema89_weekly_aligned[i]

        if position == 0:
            # LONG: Price above KAMA and weekly trend bullish
            if close[i] > kama[i] and weekly_bullish:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA and weekly trend bearish
            elif close[i] < kama[i] and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or weekly trend turns bearish
            if close[i] < kama[i] or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or weekly trend turns bullish
            if close[i] > kama[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals