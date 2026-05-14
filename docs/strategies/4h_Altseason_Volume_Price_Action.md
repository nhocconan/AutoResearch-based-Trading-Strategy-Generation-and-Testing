# Strategy: 4h_Altseason_Volume_Price_Action

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.112 | +25.0% | -20.1% | 146 | PASS |
| ETHUSDT | 0.355 | +47.8% | -14.3% | 144 | PASS |
| SOLUSDT | 1.063 | +245.0% | -30.3% | 157 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.869 | -5.9% | -8.7% | 59 | FAIL |
| ETHUSDT | 0.633 | +19.2% | -8.9% | 53 | PASS |
| SOLUSDT | 0.580 | +19.2% | -10.5% | 45 | PASS |

## Code
```python
# 4h_Altseason_Volume_Price_Action
# Hypothesis: On 4h timeframe, combine volume spikes with price action near key levels (ATR-based) and trend filter from 12h EMA to capture altseason momentum while avoiding chop.
# Long when: price > ATR-based support, volume spike, and price above 12h EMA50
# Short when: price < ATR-based resistance, volume spike, and price below 12h EMA50
# Uses volume confirmation to avoid false breakouts and trend filter to align with higher timeframe momentum.
# Targets 20-40 trades/year by requiring confluence of volume, price action, and trend.
# Works in bull markets (altseason rallies) and bear markets (bear rallies/trading ranges) by filtering with trend and volume.

name = "4h_Altseason_Volume_Price_Action"
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

    # Get 12h data for trend filter (EMA50) ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate EMA50 on 12h close
    close_12h = df_12h['close']
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate ATR(14) for dynamic support/resistance levels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Dynamic support/resistance: close ± 0.5 * ATR
    support = close - 0.5 * atr
    resistance = close + 0.5 * atr

    # Volume confirmation: current volume > 1.5x average of last 12 periods (3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(support[i]) or np.isnan(resistance[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]

        if position == 0:
            # LONG: price above support, volume spike, and above 12h EMA50
            if close[i] > support[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: price below resistance, volume spike, and below 12h EMA50
            elif close[i] < resistance[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below support OR closes below 12h EMA50
            if close[i] < support[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above resistance OR closes above 12h EMA50
            if close[i] > resistance[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 15:27
