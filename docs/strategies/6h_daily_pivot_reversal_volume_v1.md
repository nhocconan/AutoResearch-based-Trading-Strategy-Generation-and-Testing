# Strategy: 6h_daily_pivot_reversal_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.137 | +26.8% | -27.3% | 66 | PASS |
| ETHUSDT | 0.268 | +38.7% | -27.9% | 64 | PASS |
| SOLUSDT | 1.068 | +249.5% | -47.9% | 54 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.811 | -5.2% | -17.6% | 31 | FAIL |
| ETHUSDT | 0.065 | +5.7% | -20.3% | 29 | PASS |
| SOLUSDT | -1.122 | -20.6% | -27.3% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Pivot Reversal with Volume Confirmation
# Hypothesis: In ranging markets (common in 2025), price reverts to daily pivot points.
# We fade at S3/R3 (strong support/resistance) and breakout at S4/R4.
# Volume confirms institutional participation at these key levels.
# Works in both bull/bear as it captures mean reversion and breakouts.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_daily_pivot_reversal_volume_v1"
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # S1 = 2*P - H, S2 = P - (H - L), S3 = S2 - (H - L)
    # R1 = 2*P - L, R2 = P + (H - L), R3 = R2 + (H - L)
    # S4 = S3 - (H - L), R4 = R3 + (H - L)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    s1 = 2 * pivot - daily_high
    s2 = pivot - (daily_high - daily_low)
    s3 = s2 - (daily_high - daily_low)
    s4 = s3 - (daily_high - daily_low)
    r1 = 2 * pivot - daily_low
    r2 = pivot + (daily_high - daily_low)
    r3 = r2 + (daily_high - daily_low)
    r4 = r3 + (daily_high - daily_low)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: current volume > 1.8x 24-period average (institutional interest)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or breaks below S3 (stop)
            if close[i] >= r3_6h[i] or close[i] <= s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or breaks above R3 (stop)
            if close[i] <= s3_6h[i] or close[i] >= r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Fade at S3/R3: price touches level and reverses
                # Long: price touches or goes below S3 but closes back above it
                if close[i] > s3_6h[i] and low[i] <= s3_6h[i]:
                    # Additional confirmation: price closing in upper half of daily range
                    daily_range = r3_6h[i] - s3_6h[i]
                    if daily_range > 0:
                        close_position = (close[i] - s3_6h[i]) / daily_range
                        if close_position > 0.5:  # Closing in upper half
                            position = 1
                            signals[i] = 0.25
                # Short: price touches or goes above R3 but closes back below it
                elif close[i] < r3_6h[i] and high[i] >= r3_6h[i]:
                    # Additional confirmation: price closing in lower half of daily range
                    daily_range = r3_6h[i] - s3_6h[i]
                    if daily_range > 0:
                        close_position = (close[i] - s3_6h[i]) / daily_range
                        if close_position < 0.5:  # Closing in lower half
                            position = -1
                            signals[i] = -0.25
                # Breakout continuation: price breaks S4/R4 with volume
                # Long breakout: price closes above S4
                elif close[i] > s4_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below R4
                elif close[i] < r4_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 09:15
