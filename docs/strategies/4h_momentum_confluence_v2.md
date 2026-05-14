# Strategy: 4h_momentum_confluence_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.255 | +12.3% | -11.3% | 603 | FAIL |
| ETHUSDT | -0.551 | +0.6% | -12.3% | 545 | FAIL |
| SOLUSDT | 0.631 | +72.1% | -18.0% | 451 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.712 | +15.0% | -5.5% | 178 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_momentum_confluence_v2
# Hypothesis: Combines 4h momentum (price relative to SMA50) with 12h trend direction (SMA50 slope) and volume confirmation.
# Long when price > SMA50, 12h SMA50 slope > 0, volume > 1.5x average.
# Short when price < SMA50, 12h SMA50 slope < 0, volume > 1.5x average.
# Exit when momentum breaks (price crosses SMA50 opposite direction) or volume drops below average.
# Uses tight entry conditions to limit trades and reduce fee drag. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_momentum_confluence_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h SMA50 for momentum
    sma_period = 50
    close_series = pd.Series(close)
    sma50 = close_series.rolling(window=sma_period, min_periods=sma_period).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 12h data for trend direction (SMA50 slope)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    sma50_12h = pd.Series(close_12h).rolling(window=sma_period, min_periods=sma_period).mean().values
    # Calculate slope: positive if current SMA > SMA 3 periods ago
    sma50_slope_12h = np.full(len(close_12h), np.nan)
    for i in range(3, len(close_12h)):
        if not np.isnan(sma50_12h[i]) and not np.isnan(sma50_12h[i-3]):
            sma50_slope_12h[i] = sma50_12h[i] - sma50_12h[i-3]
    sma50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, sma50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(sma_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50[i]) or np.isnan(vol_ma[i]) or np.isnan(sma50_slope_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below SMA50 or volume drops below average
            if close[i] < sma50[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above SMA50 or volume drops below average
            if close[i] > sma50[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above SMA50, 12h SMA50 slope positive, volume surge
            if (close[i] > sma50[i] and 
                sma50_slope_12h_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below SMA50, 12h SMA50 slope negative, volume surge
            elif (close[i] < sma50[i] and 
                  sma50_slope_12h_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 20:26
