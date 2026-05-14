# Strategy: 4h_Keltner_Channel_Breakout_EMA21_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.075 | +13.1% | -17.9% | 99 | DISCARD |
| ETHUSDT | 0.304 | +41.6% | -17.6% | 89 | KEEP |
| SOLUSDT | 1.219 | +290.4% | -27.4% | 87 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.248 | +9.8% | -8.6% | 33 | KEEP |
| SOLUSDT | -0.013 | +3.9% | -16.3% | 31 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_EMA21_Trend_Filter
Hypothesis: Price breaking above/below 2x ATR Keltner Channel with EMA21 trend filter and volume confirmation captures strong trending moves while avoiding false breakouts. Works in bull/bear by following trend direction. Uses 4h timeframe with 1d EMA21 trend filter for higher timeframe context.
"""

name = "4h_Keltner_Channel_Breakout_EMA21_Trend_Filter"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate Keltner Channel (20-period EMA, 2x ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr

    # 1d EMA21 trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)

    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(21, n):  # Start after EMA21 warmup
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner + EMA21 uptrend + volume confirmation
            if (close[i] > kc_upper[i] and 
                close[i] > ema_21_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner + EMA21 downtrend + volume confirmation
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_21_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA21 (trend reversal)
            if close[i] < ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA21 (trend reversal)
            if close[i] > ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 14:03
