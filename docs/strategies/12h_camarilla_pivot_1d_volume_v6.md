# Strategy: 12h_camarilla_pivot_1d_volume_v6

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.280 | +2.0% | -17.2% | 378 | FAIL |
| ETHUSDT | 0.357 | +46.4% | -18.8% | 352 | PASS |
| SOLUSDT | 0.330 | +49.8% | -41.8% | 340 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.756 | -8.6% | -20.1% | 115 | FAIL |
| SOLUSDT | 0.307 | +11.2% | -15.1% | 105 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v6
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels (H3/L3) with volume confirmation.
Long when price crosses above H3 with volume > 1.5x 20-period average.
Short when price crosses below L3 with volume > 1.5x 20-period average.
Exit when price touches the opposite pivot level (L3 for longs, H3 for shorts).
Designed for 15-30 trades/year to minimize fee drag while capturing institutional levels.
Works in both bull/bear markets as pivot levels act as support/resistance regardless of trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v6"
timezone = "12h"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    # H4 = close + (high - low) * 1.1
    # L4 = close - (high - low) * 1.1
    # We'll use H3/L3 for entries and H4/L4 for stronger signals
    range_1d = high_1d - low_1d
    H3_1d = close_1d + range_1d * 1.1 / 2
    L3_1d = close_1d - range_1d * 1.1 / 2
    H4_1d = close_1d + range_1d * 1.1
    L4_1d = close_1d - range_1d * 1.1
    
    # Align pivot levels to 12h timeframe (using previous day's levels)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if pivot data not available
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches L3 (opposite level)
            if low[i] <= L3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H3 (opposite level)
            if high[i] >= H3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price crosses above H3
                if close[i] > H3_1d_aligned[i] and close[i-1] <= H3_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below L3
                elif close[i] < L3_1d_aligned[i] and close[i-1] >= L3_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 18:07
