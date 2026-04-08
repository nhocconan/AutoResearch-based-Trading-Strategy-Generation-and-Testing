# Strategy: 6h_elder_ray_12h_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.956 | -24.8% | -38.8% | 2329 | FAIL |
| ETHUSDT | -0.798 | -27.4% | -33.8% | 2320 | FAIL |
| SOLUSDT | 0.313 | +46.6% | -30.7% | 2355 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.052 | +5.8% | -11.7% | 714 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray with 12h Trend Filter v1
# Hypothesis: Elder Ray (bull/bear power) combined with 12h EMA trend filter captures momentum
# in both bull and bear markets. Bull power > 0 and bear power < 0 with trend alignment
# provides high-probability entries. Works in ranging markets via mean reversion at extremes.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Elder Ray components: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth the power signals
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bear power turns positive (momentum loss) or price below EMA13
            if bear_power_smooth[i] > 0 or close[i] < ema13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: bull power turns negative (momentum loss) or price above EMA13
            if bull_power_smooth[i] < 0 or close[i] > ema13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: bull power positive AND price above 12h EMA (uptrend)
            if bull_power_smooth[i] > 0 and close[i] > ema_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: bear power negative AND price below 12h EMA (downtrend)
            elif bear_power_smooth[i] < 0 and close[i] < ema_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 09:17
