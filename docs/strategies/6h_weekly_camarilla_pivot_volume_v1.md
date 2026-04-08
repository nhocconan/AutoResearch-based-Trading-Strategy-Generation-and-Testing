# Strategy: 6h_weekly_camarilla_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.537 | +6.5% | -8.0% | 444 | FAIL |
| ETHUSDT | -0.624 | -0.4% | -9.0% | 496 | FAIL |
| SOLUSDT | 0.673 | +72.8% | -15.1% | 420 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.688 | +14.6% | -7.4% | 233 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Camarilla Pivot with Volume Filter
# Hypothesis: Weekly Camarilla levels (R3/S3 and R4/S4) act as strong institutional barriers.
# Price breaking above R4 with volume indicates bullish continuation; breaking below S4 indicates bearish continuation.
# Price bouncing off R3/S3 with volume indicates mean reversion.
# Works in both bull and bear markets: In bull, breaks above R4 continue up; breaks below S3 get bought.
# In bear, breaks below S4 continue down; breaks above R3 get sold.
# Volume filter ensures only institutional participation triggers entries.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_weekly_camarilla_pivot_volume_v1"
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
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly Camarilla pivot points
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r3 = weekly_pivot + (weekly_range * 1.1 / 2)
    weekly_s3 = weekly_pivot - (weekly_range * 1.1 / 2)
    weekly_r4 = weekly_pivot + (weekly_range * 1.1)
    weekly_s4 = weekly_pivot - (weekly_range * 1.1)
    
    # Align to 6h timeframe (use previous week's levels)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S3 or volume drops
            if (close[i] <= weekly_s3_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R3 or volume drops
            if (close[i] >= weekly_r3_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R4 with volume
            if ((high[i] > weekly_r4_aligned[i] or high[i] > weekly_r3_aligned[i]) and 
                (close[i] > weekly_r4_aligned[i] or close[i] > weekly_r3_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume
            elif ((low[i] < weekly_s4_aligned[i] or low[i] < weekly_s3_aligned[i]) and 
                  (close[i] < weekly_s4_aligned[i] or close[i] < weekly_s3_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 10:11
