# Strategy: 6h_Elder_Ray_Power_1wTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.063 | +17.2% | -13.7% | 93 | FAIL |
| ETHUSDT | 0.087 | +23.8% | -21.6% | 94 | PASS |
| SOLUSDT | 1.207 | +190.8% | -15.7% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.430 | +10.8% | -6.7% | 24 | PASS |
| SOLUSDT | -0.114 | +4.3% | -7.3% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_Elder_Ray_Power_1wTrend_Volume
# Hypothesis: Elder Ray combines bull power (high - EMA13) and bear power (EMA13 - low) to measure trend strength.
# Long when bull power > 0 and rising, bear power < 0 and falling, with weekly uptrend and volume confirmation.
# Short when bear power < 0 and falling, bull power > 0 and rising, with weekly downtrend and volume confirmation.
# Uses weekly EMA20 for trend filter, daily EMA13 for power calculation, and volume > 1.3x 20-period average.
# Designed for 6h timeframe to avoid overtrading. Works in bull markets via strong bullish power and in bear markets via strong bearish power.

name = "6h_Elder_Ray_Power_1wTrend_Volume"
timeframe = "6h"
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
    if len(df_1w) < 20:
        return np.zeros(n)

    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Get daily data for EMA13 (used in Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)

    # Daily EMA13 for Elder Ray power calculation
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)

    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13_1d_aligned
    bear_power = ema_13_1d_aligned - low

    # Slope of power (1-period change) to detect strengthening/weakening
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    # Set first value to 0 to avoid roll artifact
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]

        if position == 0:
            # LONG: Bull power > 0 and rising, bear power < 0 and falling, weekly uptrend, volume confirmation
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                bear_power[i] < 0 and bear_power_slope[i] < 0 and
                price_above_weekly_ema and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power < 0 and falling, bull power > 0 and rising, weekly downtrend, volume confirmation
            elif (bear_power[i] > 0 and bear_power_slope[i] > 0 and 
                  bull_power[i] < 0 and bull_power_slope[i] < 0 and
                  price_below_weekly_ema and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull power turns negative or bear power turns positive
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power turns negative or bull power turns positive
            if bear_power[i] <= 0 or bull_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 16:02
