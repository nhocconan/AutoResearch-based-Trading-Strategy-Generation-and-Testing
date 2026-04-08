# Strategy: mtf_hma_rsi_atr_15m_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.372 | -6.8% | -33.7% | 39 | DISCARD |
| ETHUSDT | -0.754 | -36.4% | -51.2% | 56 | DISCARD |
| SOLUSDT | 0.892 | +194.2% | -34.4% | 38 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.160 | +7.7% | -13.6% | 14 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""Simple 1d EMA(10/30) with RSI timing. Minimal complexity."""
import numpy as np
import pandas as pd

name = "simple_ema_rsi_1d_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    n = len(close)
    close_s = pd.Series(close)
    
    ema10 = close_s.ewm(span=10, min_periods=10, adjust=False).mean().values
    ema30 = close_s.ewm(span=30, min_periods=30, adjust=False).mean().values
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    
    signals = np.zeros(n)
    SIZE = 0.25
    prev = 0.0
    
    for i in range(30, n):
        trend = 1.0 if ema10[i] > ema30[i] else -1.0
        
        if trend > 0:
            if rsi[i] < 65:  # not overbought
                signals[i] = SIZE
            else:
                signals[i] = prev  # hold but dont add
        elif trend < 0:
            if rsi[i] > 35:  # not oversold
                signals[i] = -SIZE
            else:
                signals[i] = prev
        prev = signals[i]
    
    return signals
```

## Last Updated
2026-04-07 04:13
