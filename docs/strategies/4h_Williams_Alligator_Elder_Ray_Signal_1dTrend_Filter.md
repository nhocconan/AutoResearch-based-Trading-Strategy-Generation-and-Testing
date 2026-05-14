# Strategy: 4h_Williams_Alligator_Elder_Ray_Signal_1dTrend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.168 | +27.3% | -8.8% | 293 | PASS |
| ETHUSDT | 0.085 | +23.8% | -10.6% | 271 | PASS |
| SOLUSDT | 0.862 | +102.0% | -12.1% | 215 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.835 | -0.6% | -4.7% | 89 | FAIL |
| ETHUSDT | 0.222 | +8.6% | -6.2% | 93 | PASS |
| SOLUSDT | -0.008 | +5.4% | -7.7% | 83 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_Williams_Alligator_Elder_Ray_Signal_1dTrend_Filter
# Hypothesis: Combine Williams Alligator (trend) and Elder Ray (bull/bear power) signals on 4h,
# filtered by 1d EMA50 trend. Enter long when Alligator bullish (jaw<teeth<lips) AND Bull Power > 0.
# Enter short when Alligator bearish (jaw>teeth>lips) AND Bear Power < 0.
# Exit when signals conflict or trend weakens. Uses volume confirmation to avoid false breakouts.
# Designed for 4-8 trades/year per symbol, works in both bull and bear via dual indicators.

name = "4h_Williams_Alligator_Elder_Ray_Signal_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Williams Alligator on 4h: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13

    # Volume confirmation: current volume > 1.5x average of last 6 periods (1.5 days)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]

        if position == 0:
            # LONG: Alligator bullish AND Bull Power positive AND volume AND uptrend
            if (jaw[i] < teeth[i] < lips[i]) and (bull_power[i] > 0) and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish AND Bear Power negative AND volume AND downtrend
            elif (jaw[i] > teeth[i] > lips[i]) and (bear_power[i] < 0) and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish OR Bull Power negative OR trend down
            if not (jaw[i] < teeth[i] < lips[i]) or (bull_power[i] <= 0) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish OR Bear Power positive OR trend up
            if not (jaw[i] > teeth[i] > lips[i]) or (bear_power[i] >= 0) or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 15:36
