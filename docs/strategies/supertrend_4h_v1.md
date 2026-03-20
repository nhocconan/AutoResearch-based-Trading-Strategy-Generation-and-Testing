# Strategy: supertrend_4h_v1

## Status
ACTIVE - Sharpe=0.197 | Return=+45.0% | DD=-32.9%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.039 | +7.1% | -20.8% | 209 |
| ETHUSDT | 0.098 | +16.7% | -32.9% | 209 |
| SOLUSDT | 0.533 | +111.4% | -44.9% | 205 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.312 | -1.4% | -11.7% | 63 |
| ETHUSDT | -0.108 | -1.0% | -23.0% | 63 |
| SOLUSDT | 0.315 | +12.2% | -24.3% | 61 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #001 - Supertrend(4h) Trend Following Strategy
==========================================================
Hypothesis: Supertrend on 4h timeframe will capture major crypto trends with fewer 
whipsaws than 1h EMA crossover. The ATR-based stop adapts to volatility, reducing 
drawdown during high-volatility periods while maintaining trend exposure.

Key improvements over baseline:
- Volatility-adjusted stops (ATR) vs fixed EMA periods
- 4h timeframe = cleaner trends, fewer false signals
- Same conservative position sizing (0.35) to control DD
- Discrete signal levels to minimize churning costs
"""

import numpy as np
import pandas as pd

name = "supertrend_4h_v1"
timeframe = "4h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # ATR(10) with proper min_periods
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    multiplier = 3.0
    period = 10
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate Supertrend with proper state tracking
    supertrend = np.zeros(n)
    trend_direction = np.zeros(n)  # 1 = long, -1 = short
    
    # Initialize first valid supertrend
    first_valid = period
    supertrend[first_valid] = upper_band[first_valid]
    trend_direction[first_valid] = -1  # Start with short bias
    
    for i in range(first_valid + 1, n):
        if np.isnan(atr[i]):
            supertrend[i] = supertrend[i-1]
            trend_direction[i] = trend_direction[i-1]
            continue
            
        # If previous trend was long
        if trend_direction[i-1] == 1:
            if close[i] > supertrend[i-1]:
                # Stay long, use lower band
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                trend_direction[i] = 1
            else:
                # Flip to short
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
        else:
            # Previous trend was short
            if close[i] < supertrend[i-1]:
                # Stay short, use upper band
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                trend_direction[i] = -1
            else:
                # Flip to long
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
    
    # Generate signals with discrete position sizing
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size - critical for drawdown control
    
    for i in range(first_valid, n):
        if trend_direction[i] == 1:
            signals[i] = SIZE
        elif trend_direction[i] == -1:
            signals[i] = -SIZE
    
    return signals
```

## Last Updated
2026-03-21 05:37
