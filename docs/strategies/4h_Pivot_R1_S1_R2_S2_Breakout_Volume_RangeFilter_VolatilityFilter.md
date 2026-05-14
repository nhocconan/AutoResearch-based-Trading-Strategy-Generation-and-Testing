# Strategy: 4h_Pivot_R1_S1_R2_S2_Breakout_Volume_RangeFilter_VolatilityFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.498 | +57.9% | -12.0% | 105 | PASS |
| ETHUSDT | 0.594 | +79.2% | -17.0% | 90 | PASS |
| SOLUSDT | 0.406 | +65.8% | -44.2% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.392 | -12.9% | -24.0% | 33 | FAIL |
| ETHUSDT | 0.552 | +17.5% | -13.3% | 37 | PASS |
| SOLUSDT | 0.460 | +16.1% | -16.2% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels (1d is HTF for 4h)
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot levels (classic floor trader pivots)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    r2_aligned = align_htf_to_ltf(prices, daily, r2)
    s2_aligned = align_htf_to_ltf(prices, daily, s2)
    
    # Volume filter: current 4h volume > 1.3x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Range filter: avoid trading when price is within 0.3% of pivot
    price_to_pivot = np.abs(close - pivot_aligned) / pivot_aligned
    range_filter = price_to_pivot > 0.003
    
    # Bollinger Band width filter to avoid choppy markets (20-period)
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + (2 * bb_std)
    bb_lower = bb_ma - (2 * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_ma
    # Only trade when BB width is above 20th percentile (avoid low volatility chop)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values
    volatility_filter = bb_width_percentile > 0.2
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter, range filter, and volatility filter all pass
        if volume_filter[i] and range_filter[i] and volatility_filter[i]:
            # Long conditions: price breaks above R2 with volume (strong breakout)
            if close[i] > r2_aligned[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from S1 with volume (above S1, below S2)
            elif close[i] > s1_aligned[i] and close[i] < s2_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S2 with volume (strong breakout)
            elif close[i] < s2_aligned[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at R1 with volume (below R1, above R2)
            elif close[i] < r1_aligned[i] and close[i] > r2_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Pivot_R1_S1_R2_S2_Breakout_Volume_RangeFilter_VolatilityFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-15 08:44
