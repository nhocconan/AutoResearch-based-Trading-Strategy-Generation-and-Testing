# Strategy: 4h_Trix_Volume_Spike_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.022 | +17.6% | -13.8% | 154 | FAIL |
| ETHUSDT | 0.131 | +26.3% | -15.2% | 143 | PASS |
| SOLUSDT | 1.020 | +185.4% | -20.0% | 126 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.208 | +8.8% | -10.0% | 56 | PASS |
| SOLUSDT | 0.442 | +14.0% | -10.7% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Trix_Volume_Spike_Trend_Filter
# Hypothesis: TRIX (triple-smoothed EMA) identifies momentum shifts. Long when TRIX > 0 with volume spike and bullish trend (price > EMA50). Short when TRIX < 0 with volume spike and bearish trend (price < EMA50). Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades. Designed for fewer trades (target: 20-40/year) to minimize fee drag and improve generalization.

name = "4h_Trix_Volume_Spike_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # TRIX: triple EMA of close, then percent change
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values

    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # percent change

    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: >2x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX > 0 (bullish momentum) + price > EMA50 (uptrend) + volume spike
            if (trix[i] > 0 and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX < 0 (bearish momentum) + price < EMA50 (downtrend) + volume spike
            elif (trix[i] < 0 and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative or price breaks below EMA50
            if (trix[i] < 0 or close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive or price breaks above EMA50
            if (trix[i] > 0 or close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 01:05
