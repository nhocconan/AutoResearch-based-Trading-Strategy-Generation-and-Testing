# Strategy: 6h_ElderRay_BullPower_Trend_1dEMA

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.033 | +16.8% | -14.0% | 1368 | DISCARD |
| ETHUSDT | 0.055 | +20.6% | -14.3% | 1343 | KEEP |
| SOLUSDT | 0.388 | +58.4% | -28.3% | 1365 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.480 | +14.3% | -8.6% | 435 | KEEP |
| SOLUSDT | 0.015 | +4.9% | -16.8% | 421 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullPower_Trend_1dEMA"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6h Elder Ray Bull Power with 1d EMA trend filter.
    - Long: Bull Power > 0 and close > 1d EMA(34)
    - Short: Bear Power < 0 and close < 1d EMA(34)
    - Exit: Opposite signal or power crosses zero
    - Uses 13-period EMA for Bull/Bear Power calculation
    - Target: 12-30 trades/year on 6h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive and above 1d EMA trend
            if bull_power[i] > 0 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative and below 1d EMA trend
            elif bear_power[i] < 0 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power becomes negative or price breaks below trend
            if bear_power[i] < 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power becomes positive or price breaks above trend
            if bull_power[i] > 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 08:55
