# Strategy: 4h_volatility_breakout_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.308 | +31.3% | -6.5% | 354 | PASS |
| ETHUSDT | 0.045 | +22.3% | -6.6% | 333 | PASS |
| SOLUSDT | 0.279 | +37.7% | -22.0% | 306 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.367 | -2.1% | -4.7% | 117 | FAIL |
| ETHUSDT | 0.630 | +12.8% | -6.4% | 111 | PASS |
| SOLUSDT | 0.836 | +15.9% | -6.2% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_volatility_breakout_volume_v2
# Hypothesis: Volatility breakouts with volume confirmation on 4h timeframe. 
# Long when price breaks above upper Bollinger Band (20,2) with volume > 1.5x average.
# Short when price breaks below lower Bollinger Band with volume > 1.5x average.
# Exit when price returns to middle Bollinger Band or volume drops below average.
# Uses Bollinger Bands from 4h timeframe, volume for confirmation.
# Target: 20-50 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd

name = "4h_volatility_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) - calculated on 4h data
    bb_period = 20
    bb_std = 2
    
    # Calculate rolling mean and std
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(bb_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below middle BB or volume drops below average
            if close[i] < bb_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above middle BB or volume drops below average
            if close[i] > bb_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper BB with volume surge
            if (close[i] > bb_upper[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower BB with volume surge
            elif (close[i] < bb_lower[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 19:42
