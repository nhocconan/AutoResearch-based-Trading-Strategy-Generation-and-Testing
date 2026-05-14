# Strategy: 4h_GoldenRatio_PriceAction_With_Volume_Confirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.275 | +13.8% | -7.8% | 95 | FAIL |
| ETHUSDT | 0.048 | +22.6% | -7.1% | 77 | PASS |
| SOLUSDT | -0.275 | +5.2% | -28.2% | 63 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.197 | +7.6% | -3.4% | 35 | PASS |

## Code
```python
# 4h_GoldenRatio_PriceAction_With_Volume_Confirmation
# Hypothesis: Price reactions at Fibonacci-derived levels (0.618, 1.618) from recent swings
# provide high-probability reversal signals. Uses 4h trend filter (price above/below 200 EMA)
# to align with higher timeframe momentum. Volume spike confirms institutional participation.
# Designed to work in both bull and bear markets by following the 4h trend direction.
# Targets low-frequency, high-quality setups to minimize fee drag.

name = "4h_GoldenRatio_PriceAction_With_Volume_Confirmation"
timeframe = "4h"
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

    # Get daily data for swing high/low calculation (more stable than 4h)
    df_1d = get_htf_data(prices, '1d')

    # Calculate 4-period swing highs and lows on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Find swing highs: higher high than previous and next bar
    swing_high = np.zeros_like(high_1d, dtype=bool)
    swing_low = np.zeros_like(low_1d, dtype=bool)
    for i in range(2, len(high_1d)-2):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and \
           high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]:
            swing_high[i] = True
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and \
           low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]:
            swing_low[i] = True

    # Get most recent swing high and low
    last_swing_high = np.full_like(high_1d, np.nan)
    last_swing_low = np.full_like(low_1d, np.nan)
    last_high = np.nan
    last_low = np.nan
    for i in range(len(high_1d)):
        if swing_high[i]:
            last_high = high_1d[i]
        if swing_low[i]:
            last_low = low_1d[i]
        last_swing_high[i] = last_high
        last_swing_low[i] = last_low

    # Calculate Fibonacci levels: 0.618 retracement and 1.618 extension
    rng = last_swing_high - last_swing_low
    fib_618 = last_swing_low + 0.618 * rng  # 61.8% retracement
    fib_1618 = last_swing_high + 0.618 * rng  # 161.8% extension

    # Align to 4h timeframe
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    fib_1618_aligned = align_htf_to_ltf(prices, df_1d, fib_1618)
    last_swing_high_aligned = align_htf_to_ltf(prices, df_1d, last_swing_high)
    last_swing_low_aligned = align_htf_to_ltf(prices, df_1d, last_swing_low)

    # 4h EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Volume spike: volume > 2.0 * 20-period average (~5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(fib_618_aligned[i]) or 
            np.isnan(fib_1618_aligned[i]) or
            np.isnan(ema200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price at 61.8% retracement support + volume spike
            if close[i] > ema200[i] and close[i] <= fib_618_aligned[i] * 1.001 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price at 61.8% retracement resistance + volume spike
            elif close[i] < ema200[i] and close[i] >= fib_618_aligned[i] * 0.999 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches 161.8% extension or trend turns bearish
            if close[i] >= fib_1618_aligned[i] * 0.999 or close[i] < ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches 161.8% extension or trend turns bullish
            if close[i] <= fib_1618_aligned[i] * 1.001 or close[i] > ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 03:50
