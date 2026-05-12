#!/usr/bin/env python3
"""
1d_PriceAction_WeeklyTrend_Filter
Hypothesis: On daily timeframe, price action relative to weekly trend provides
robust signals in both bull and bear markets. Long when daily close above
weekly EMA20 and bullish engulfing candle; short when below weekly EMA20 and
bearish engulfing candle. Uses weekly trend filter to avoid counter-trend trades.
Designed for 10-25 trades/year to minimize fee drag while capturing sustained moves.
"""

name = "1d_PriceAction_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get weekly data (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)

    # Calculate weekly EMA20
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        ema20 = ema20_weekly_aligned[i]

        if np.isnan(ema20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                         (close[i] >= open_price[i-1]) and (open_price[i] <= close[i-1])
        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                         (open_price[i] >= close[i-1]) and (close[i] <= open_price[i-1])

        if position == 0:
            # LONG: Price above weekly EMA20 + bullish engulfing
            if close[i] > ema20 and bullish_engulf:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly EMA20 + bearish engulfing
            elif close[i] < ema20 and bearish_engulf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly EMA20
            if close[i] < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly EMA20
            if close[i] > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals