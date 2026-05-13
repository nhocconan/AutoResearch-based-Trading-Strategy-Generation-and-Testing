# 1d_Engulfing_1wTrend
# Hypothesis: Engulfing patterns on daily chart combined with weekly trend filter
# to capture high-probability reversals at trend extremes. Works in both bull and bear
# by fading extreme moves when weekly trend shows exhaustion.
# Target: 10-20 trades/year per symbol (low frequency, high quality)

name = "1d_Engulfing_1wTrend"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 20-period EMA on weekly data for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Placeholder, will fix below

    # Actually calculate properly:
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Still wrong

    # Correct calculation:
    prev_close = np.roll(close, 1)
    prev_open = np.roll(open_price, 1)
    bullish_engulf = (close > open_price) & (prev_close < prev_open) & \
                     (close >= prev_open) & (open_price <= prev_close)
    bearish_engulf = (close < open_price) & (prev_close > prev_open) & \
                     (close <= prev_open) & (open_price >= prev_close)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bullish engulfing at weekly support (price below weekly EMA = potential bounce)
            if bullish_engulf[i] and close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish engulfing at weekly resistance (price above weekly EMA = potential rejection)
            elif bearish_engulf[i] and close[i] > ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish engulfing or price crosses above weekly EMA (trend resumption)
            if bearish_engulf[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish engulfing or price crosses below weekly EMA (trend resumption)
            if bullish_engulf[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals