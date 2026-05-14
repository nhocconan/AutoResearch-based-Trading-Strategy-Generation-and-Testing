# Strategy: 4h_Keltner_Breakout_1dTrend_Volume_Momentum

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.199 | +29.2% | -8.8% | 124 | PASS |
| ETHUSDT | 0.002 | +19.0% | -15.0% | 125 | PASS |
| SOLUSDT | 0.704 | +90.8% | -19.4% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.790 | -0.7% | -5.7% | 43 | FAIL |
| ETHUSDT | 0.720 | +17.0% | -6.5% | 40 | PASS |
| SOLUSDT | 0.332 | +10.5% | -10.4% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_1dTrend_Volume_Momentum
# Hypothesis: Price breaking above/below Keltner Channel (ATR-based) with 1-day EMA50 trend filter and volume momentum captures strong moves. Keltner adapts to volatility, reducing false breakouts in ranging markets. Trend filter ensures alignment with higher timeframe direction. Works in bull/bear by following 1d trend. Target: 20-40 trades/year.

name = "4h_Keltner_Breakout_1dTrend_Volume_Momentum"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Keltner Channel (20, 2.0) on 4h
    # Middle = EMA20 of close
    # Upper = Middle + 2 * ATR(20)
    # Lower = Middle - 2 * ATR(20)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values  # Simple TR approximation
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr

    # Volume momentum: current > 1.5x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_momentum = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(volume_momentum[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner Upper + 1d EMA50 uptrend + volume momentum
            if (close[i] > kc_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_momentum[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner Lower + 1d EMA50 downtrend + volume momentum
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_momentum[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Keltner Middle (reversion to mean)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Keltner Middle (reversion to mean)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 14:26
