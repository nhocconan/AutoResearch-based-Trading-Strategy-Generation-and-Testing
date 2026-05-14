# Strategy: 6h_RSI_34_60_Trend_1d

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.146 | +12.9% | -11.0% | 188 | FAIL |
| ETHUSDT | 0.033 | +20.5% | -14.5% | 199 | PASS |
| SOLUSDT | -0.347 | -8.3% | -28.7% | 193 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.300 | +10.2% | -8.6% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "6h_RSI_34_60_Trend_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # RSI(34) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI between 34-60 in uptrend with volume
            if 34 <= rsi[i] <= 60 and close[i] > ema_34_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI between 40-66 in downtrend with volume
            elif 40 <= rsi[i] <= 66 and close[i] < ema_34_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI reverts to 50 or trend fails
            if position == 1 and (rsi[i] >= 60 or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi[i] <= 40 or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 05:18
