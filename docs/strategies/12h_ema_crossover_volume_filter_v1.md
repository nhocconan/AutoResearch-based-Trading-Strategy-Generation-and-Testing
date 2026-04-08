# Strategy: 12h_ema_crossover_volume_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.116 | +14.5% | -11.4% | 32 | FAIL |
| ETHUSDT | -0.931 | -22.1% | -29.1% | 32 | FAIL |
| SOLUSDT | 1.148 | +185.6% | -17.8% | 39 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.018 | +5.3% | -14.3% | 9 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_ema_crossover_volume_filter_v1
Hypothesis: On 12-hour timeframe, use EMA(10)/EMA(30) crossover with volume confirmation to capture medium-term trends while avoiding whipsaws. The 12h timeframe balances responsiveness with reduced noise, and volume filters ensure institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_crossover_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA periods
    ema_fast = 10
    ema_slow = 30
    
    # Calculate EMAs
    close_series = pd.Series(close)
    ema_fast_values = close_series.ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean().values
    ema_slow_values = close_series.ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean().values
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(ema_slow, 20), n):
        # Skip if data not available
        if (np.isnan(ema_fast_values[i]) or np.isnan(ema_slow_values[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: EMA bearish crossover
            if ema_fast_values[i] <= ema_slow_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA bullish crossover
            if ema_fast_values[i] >= ema_slow_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish crossover: fast EMA crosses above slow EMA
                if ema_fast_values[i] > ema_slow_values[i] and ema_fast_values[i-1] <= ema_slow_values[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Bearish crossover: fast EMA crosses below slow EMA
                elif ema_fast_values[i] < ema_slow_values[i] and ema_fast_values[i-1] >= ema_slow_values[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 19:04
