# Strategy: 6h_ma_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.449 | -1.6% | -5.3% | 131 | FAIL |
| ETHUSDT | -0.432 | +12.9% | -4.2% | 102 | FAIL |
| SOLUSDT | 0.137 | +26.1% | -7.5% | 81 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.165 | +7.1% | -3.0% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_ma_trend_volume_v1
# Hypothesis: 60-period moving average trend filter with volume confirmation on 6h timeframe.
# Long when price crosses above 60-period MA with volume > 1.5x average, short when price crosses below.
# Exit on opposite cross or when volume drops below average.
# Designed to capture sustained trends in both bull and bear markets with volume confirmation to reduce whipsaw.
# Target: 80-160 total trades over 4 years (~20-40/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ma_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 60-period moving average
    ma_60 = pd.Series(close).rolling(window=60, min_periods=60).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ma_60[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below MA or volume drops below average
            if close[i] < ma_60[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above MA or volume drops below average
            if close[i] > ma_60[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # MA crossover entries: price crosses above MA (long) or below MA (short)
            if (close[i] > ma_60[i]) and volume_ok:
                # Additional confirmation: previous close was below MA to confirm crossover
                if i > 0 and close[i-1] <= ma_60[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif (close[i] < ma_60[i]) and volume_ok:
                # Additional confirmation: previous close was above MA to confirm crossover
                if i > 0 and close[i-1] >= ma_60[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 14:24
