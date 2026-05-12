#!/usr/bin/env python3
# 12h_ThreeWhiteSoldiers_ThreeBlackCrows_Trend_Volume
# Hypothesis: Use 12h Three White Soldiers / Three Black Crows patterns for trend continuation entries.
# Enter long on Three White Soldiers with volume confirmation and 1d uptrend.
# Enter short on Three Black Crows with volume confirmation and 1d downtrend.
# Exit when pattern breaks or trend reverses.
# This strategy captures momentum continuation with strict pattern recognition to limit trades.
# Works in bull markets (continuation of uptrends) and bear markets (continuation of downtrends).
# Targets 15-25 trades/year by requiring rare candlestick patterns.

name = "12h_ThreeWhiteSoldiers_ThreeBlackCrows_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5x average of last 2 periods (1 day)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    # Three White Soldiers: three consecutive bullish candles with higher closes
    bullish = close > open_
    higher_close = close > np.roll(close, 1)
    three_white_soldiers = bullish & np.roll(bullish, 1) & np.roll(bullish, 2) & \
                           higher_close & np.roll(higher_close, 1) & np.roll(higher_close, 2)

    # Three Black Crows: three consecutive bearish candles with lower closes
    bearish = close < open_
    lower_close = close < np.roll(close, 1)
    three_black_crows = bearish & np.roll(bearish, 1) & np.roll(bearish, 2) & \
                        lower_close & np.roll(lower_close, 1) & np.roll(lower_close, 2)

    # Align daily trend to 12h timeframe
    price_above_ema = close > ema_34_aligned
    price_below_ema = close < ema_34_aligned

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Three White Soldiers with volume and uptrend
            if three_white_soldiers[i] and volume_ok[i] and price_above_ema[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Three Black Crows with volume and downtrend
            elif three_black_crows[i] and volume_ok[i] and price_below_ema[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: pattern breaks or trend turns down
            if not three_white_soldiers[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: pattern breaks or trend turns up
            if not three_black_crows[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals