# Strategy: 4H_Camarilla_R1_S1_Breakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.260 | +28.4% | -11.0% | 329 | PASS |
| ETHUSDT | 0.656 | +45.1% | -6.9% | 297 | PASS |
| SOLUSDT | -0.162 | +10.6% | -16.2% | 257 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.703 | -3.9% | -4.8% | 128 | FAIL |
| ETHUSDT | 1.376 | +21.2% | -4.8% | 114 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot R1/S1 breakout on 4h, filtered by 12h trend and volume spikes.
# Uses tight entries (Camarilla levels act as strong support/resistance) with trend alignment
# to capture breakouts in trending markets while avoiding false signals in ranges.
# Works in bull/bear by following 12h trend direction; volume confirms institutional interest.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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

    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')

    # Calculate Camarilla pivot levels for 4h (based on previous 4h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous bar's high/low/close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12

    # Trend filter: 12h EMA50
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: current volume > 2.0 x 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or 
            np.isnan(s1[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 in uptrend with volume spike
            if (close[i] > r1[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 in downtrend with volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend turns down
            if close[i] < s1[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend turns up
            if close[i] > r1[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 04:28
