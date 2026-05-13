# 1D_GoldenRatio_PriceAction_With_Volume_Confirmation
# Hypothesis: Price reactions at Fibonacci-derived levels (0.618, 1.618) from weekly swing points
# provide high-probability reversal signals. Uses weekly trend filter (price above/below 50 EMA)
# to align with higher timeframe momentum. Volume spike confirms institutional participation.
# Designed to work in both bull and bear markets by following the weekly trend direction.
# Targets low-frequency, high-quality setups to minimize fee drag.

name = "1D_GoldenRatio_PriceAction_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for swing high/low calculation (more stable than daily)
    df_1w = get_htf_data(prices, '1w')

    # Calculate 4-period swing highs and lows on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Find swing highs: higher high than previous and next bar
    swing_high = np.zeros_like(high_1w, dtype=bool)
    swing_low = np.zeros_like(low_1w, dtype=bool)
    for i in range(2, len(high_1w)-2):
        if high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and \
           high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]:
            swing_high[i] = True
        if low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and \
           low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]:
            swing_low[i] = True

    # Get most recent swing high and low
    last_swing_high = np.full_like(high_1w, np.nan)
    last_swing_low = np.full_like(low_1w, np.nan)
    last_high = np.nan
    last_low = np.nan
    for i in range(len(high_1w)):
        if swing_high[i]:
            last_high = high_1w[i]
        if swing_low[i]:
            last_low = low_1w[i]
        last_swing_high[i] = last_high
        last_swing_low[i] = last_low

    # Calculate Fibonacci levels: 0.618 retracement and 1.618 extension
    rng = last_swing_high - last_swing_low
    fib_618 = last_swing_low + 0.618 * rng  # 61.8% retracement
    fib_1618 = last_swing_high + 0.618 * rng  # 161.8% extension

    # Align to daily timeframe
    fib_618_aligned = align_htf_to_ltf(prices, df_1w, fib_618)
    fib_1618_aligned = align_htf_to_ltf(prices, df_1w, fib_1618)
    last_swing_high_aligned = align_htf_to_ltf(prices, df_1w, last_swing_high)
    last_swing_low_aligned = align_htf_to_ltf(prices, df_1w, last_swing_low)

    # Weekly EMA50 for trend filter
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)

    # Volume spike: volume > 2.0 * 20-period average (~20 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(fib_618_aligned[i]) or 
            np.isnan(fib_1618_aligned[i]) or
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price at 61.8% retracement support + volume spike
            if close[i] > ema50_aligned[i] and close[i] <= fib_618_aligned[i] * 1.001 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price at 61.8% retracement resistance + volume spike
            elif close[i] < ema50_aligned[i] and close[i] >= fib_618_aligned[i] * 0.999 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches 161.8% extension or trend turns bearish
            if close[i] >= fib_1618_aligned[i] * 0.999 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches 161.8% extension or trend turns bullish
            if close[i] <= fib_1618_aligned[i] * 1.001 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals