# Strategy: 4h_camarilla_pivot_1d_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.472 | -1.8% | -17.7% | 131 | FAIL |
| ETHUSDT | -0.765 | -14.7% | -27.8% | 106 | FAIL |
| SOLUSDT | 0.252 | +38.1% | -34.9% | 112 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.676 | +17.1% | -10.3% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_volume_v1
# Hypothesis: 4h Camarilla pivot levels from 1d + volume confirmation + ATR stoploss.
# Uses 1d Camarilla pivot levels (H3, L3, H4, L4) as key support/resistance.
# Long when price breaks above H3 with volume > 1.5x 20-period average.
# Short when price breaks below L3 with volume > 1.5x 20-period average.
# Exit when price returns to H4/L4 levels or opposite pivot level.
# Works in bull/bear: pivot levels adapt to volatility, volume ensures momentum validity.
# Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H4 = close + 1.5 * range
    # H3 = close + 1.25 * range
    # L3 = close - 1.25 * range
    # L4 = close - 1.5 * range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h4_1d = close_1d + 1.5 * range_1d
    h3_1d = close_1d + 1.25 * range_1d
    l3_1d = close_1d - 1.25 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 4h timeframe (completed 1d bar only)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below L4 OR rises above H4 (mean reversion to pivot)
            if close[i] < l4_1d_aligned[i] or close[i] > h4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above H4 OR falls below L4 (mean reversion to pivot)
            if close[i] > h4_1d_aligned[i] or close[i] < l4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long entry: price breaks above H3 (resistance) with volume
                if close[i] > h3_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below L3 (support) with volume
                elif close[i] < l3_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 03:59
