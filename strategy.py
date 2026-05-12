#!/usr/bin/env python3
# 1d_Price_Action_With_Weekly_Trend_and_Volume
# Hypothesis: Combines daily price action (close > open for bullish, close < open for bearish) with weekly trend filter (price above/below weekly EMA20) and volume confirmation (>1.5x 20-day average volume). Works in bull markets via buying dips in uptrend and in bear markets via selling rallies in downtrend. Uses only 3 conditions to minimize overtrading and maximize edge.

name = "1d_Price_Action_With_Weekly_Trend_and_Volume"
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
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]

        # Bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]

        if position == 0:
            # LONG: Bullish candle, price above weekly EMA, volume confirmation
            if bullish_candle and price_above_weekly_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish candle, price below weekly EMA, volume confirmation
            elif bearish_candle and price_below_weekly_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish candle or price breaks below weekly EMA
            if bearish_candle or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish candle or price breaks above weekly EMA
            if bullish_candle or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals