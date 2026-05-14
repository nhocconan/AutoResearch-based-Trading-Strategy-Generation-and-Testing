# Strategy: 4h_1d_1w_Trend_Reversal

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.405 | +41.8% | -8.8% | 125 | PASS |
| ETHUSDT | 0.250 | +34.4% | -17.1% | 120 | PASS |
| SOLUSDT | 1.082 | +177.0% | -19.0% | 164 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.994 | -1.5% | -6.5% | 42 | FAIL |
| ETHUSDT | 0.225 | +8.7% | -7.8% | 36 | PASS |
| SOLUSDT | 0.018 | +5.7% | -8.7% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Trend_Reversal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # Calculate 30-period EMA on 1w close for higher timeframe trend
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_30_1d_aligned[i]) or np.isnan(ema_30_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_30_1d_aligned[i]
        ema_1w = ema_30_1w_aligned[i]
        
        if position == 0:
            # Enter long: Price above both EMAs (bullish alignment)
            if close[i] > ema_1d and close[i] > ema_1w:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below both EMAs (bearish alignment)
            elif close[i] < ema_1d and close[i] < ema_1w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below either EMA
            if close[i] < ema_1d or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above either EMA
            if close[i] > ema_1d or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 03:38
