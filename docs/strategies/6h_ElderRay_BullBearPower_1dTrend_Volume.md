# Strategy: 6h_ElderRay_BullBearPower_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.248 | +31.7% | -10.1% | 226 | PASS |
| ETHUSDT | 0.516 | +51.7% | -14.9% | 182 | PASS |
| SOLUSDT | 1.114 | +169.1% | -22.1% | 152 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.777 | -1.8% | -7.9% | 89 | FAIL |
| ETHUSDT | 0.653 | +16.4% | -7.8% | 63 | PASS |
| SOLUSDT | 0.237 | +9.1% | -12.8% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray's Bull Power (High - EMA13) and Bear Power (EMA13 - Low) capture institutional buying/selling pressure.
# Combined with 1-day EMA34 trend filter and volume spikes, it identifies strong directional moves with institutional backing.
# Works in bull markets via Bull Power > 0 + uptrend, and in bear markets via Bear Power > 0 + downtrend.
# Volume spike confirms institutional participation. Target: 15-25 trades/year.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Elder Ray components on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = ema_13 - low   # Bear Power: EMA13 - Low

    # Volume spike: current > 2.0x average of last 24 bars (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (buying pressure) + 1d EMA34 uptrend + volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (selling pressure) + 1d EMA34 downtrend + volume spike
            elif (bear_power[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative (loss of buying pressure) OR close below EMA13
            if (bull_power[i] <= 0 or close[i] < ema_13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative (loss of selling pressure) OR close above EMA13
            if (bear_power[i] <= 0 or close[i] > ema_13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 14:32
