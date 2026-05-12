#!/usr/bin/env python3
"""
1d_WeeklyTrend_DailyReversion
Hypothesis: On 1d timeframe, take mean-reversion trades when price deviates from weekly trend (EMA21) with RSI confirmation. In bull markets, buy dips to weekly EMA; in bear markets, sell rallies to weekly EMA. Uses weekly EMA21 as dynamic support/resistance and daily RSI for entry timing. Designed for low trade frequency (10-30 trades/year) to minimize fee drag while capturing meaningful reversals in both bull and bear regimes.
"""

name = "1d_WeeklyTrend_DailyReversion"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Weekly EMA21 for trend direction (using weekly close)
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Align weekly EMA21 to daily timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Daily RSI for entry timing (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(21, n):
        # Skip if weekly EMA not available
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below weekly EMA (dip in uptrend) + RSI oversold
            if close[i] < ema21_1w_aligned[i] and rsi_values[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: Price above weekly EMA (rally in downtrend) + RSI overbought
            elif close[i] > ema21_1w_aligned[i] and rsi_values[i] > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back above weekly EMA OR RSI overbought
            if close[i] > ema21_1w_aligned[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back below weekly EMA OR RSI oversold
            if close[i] < ema21_1w_aligned[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals