# Strategy: 6H_ELDER_RAY_POWER_1D_TREND_FILTER

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.425 | +49.3% | -14.2% | 116 | PASS |
| ETHUSDT | 0.138 | +26.5% | -14.1% | 125 | PASS |
| SOLUSDT | 0.915 | +190.6% | -32.2% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.101 | +4.0% | -5.8% | 42 | FAIL |
| ETHUSDT | 0.541 | +17.1% | -8.3% | 35 | PASS |
| SOLUSDT | 0.435 | +15.3% | -10.6% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6H_ELDER_RAY_POWER_1D_TREND_FILTER
# Hypothesis: Elder Ray (Bull/Bear power) measures bull/bear strength relative to EMA13.
# In 1d uptrend (EMA34), go long when Bull Power > 0 and rising; in downtrend, go short when Bear Power < 0 and falling.
# Works in both bull and bear markets: trend filter avoids counter-trend trades, Elder Ray captures momentum within trend.
# Target: 15-25 trades/year on 6h timeframe.

name = "6H_ELDER_RAY_POWER_1D_TREND_FILTER"
timeframe = "6h"
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
    
    # Daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Rising/falling power detection (1-bar change)
    bull_power_rising = bull_power_aligned > np.roll(bull_power_aligned, 1)
    bear_power_falling = bear_power_aligned < np.roll(bear_power_aligned, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + Bull Power > 0 and rising
            if (close[i] > ema34_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                bull_power_rising[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + Bear Power < 0 and falling
            elif (close[i] < ema34_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  bear_power_falling[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or power weakening
            if (close[i] <= ema34_aligned[i] or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or power weakening
            if (close[i] >= ema34_aligned[i] or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 09:54
