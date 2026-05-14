# Strategy: 4h_1D_Camarilla_R3S3_Breakout_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.590 | +50.2% | -11.9% | 272 | PASS |
| ETHUSDT | 0.379 | +42.4% | -12.7% | 258 | PASS |
| SOLUSDT | 0.653 | +88.7% | -26.3% | 233 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.625 | -0.5% | -7.7% | 108 | FAIL |
| ETHUSDT | 1.391 | +30.9% | -8.7% | 96 | PASS |
| SOLUSDT | 0.737 | +17.9% | -11.6% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1D_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Breakout at daily Camarilla R3/S3 levels with volume confirmation and 1d trend filter.
# Uses 1d timeframe for Camarilla levels and trend filter, 4h for entry/exit.
# Designed to work in both bull and bear markets by requiring volume confirmation and trend alignment.
# Targets 20-50 trades/year on 4h timeframe to avoid fee drag.

name = "4h_1D_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "4h"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Camarilla R3 and S3 levels from previous 1d OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Volume confirmation: current volume > 1.8x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R3 with bullish trend and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Break below Camarilla S3 with bearish trend and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns bearish
            if close[i] < camarilla_r3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns bullish
            if close[i] > camarilla_s3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
```

## Last Updated
2026-05-12 16:35
