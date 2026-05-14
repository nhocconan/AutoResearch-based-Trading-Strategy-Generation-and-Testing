# Strategy: 4h_Williams_Alligator_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.255 | +33.9% | -11.1% | 218 | PASS |
| ETHUSDT | 0.107 | +24.6% | -14.5% | 220 | PASS |
| SOLUSDT | 1.078 | +199.1% | -23.7% | 202 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.992 | -5.0% | -8.9% | 80 | FAIL |
| ETHUSDT | 0.214 | +8.9% | -12.0% | 69 | PASS |
| SOLUSDT | -0.263 | -0.3% | -13.3% | 76 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "4h_Williams_Alligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Williams Alligator (Williams SMMA)
    def smma(arr, period):
        sma = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return sma
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips < Teeth (death cross) or 1d trend down
            if lips[i] < teeth[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips > Teeth (golden cross) or 1d trend up
            if lips[i] > teeth[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 20:32
