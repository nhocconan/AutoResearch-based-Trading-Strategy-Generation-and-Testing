# Strategy: 4h_12h_camarilla_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.216 | +6.6% | -18.8% | 471 | FAIL |
| ETHUSDT | -0.625 | -19.9% | -41.8% | 479 | FAIL |
| SOLUSDT | 0.448 | +66.5% | -39.8% | 412 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.401 | +13.0% | -13.6% | 139 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels (R3, S3 - stronger reversal points)
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4
    # R4/S4 - breakout levels
    r4 = pivot + (prev_high - prev_low) * 1.1 / 2
    s4 = pivot - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: 4h volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if pivot data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or touches S4 (strong support/breakdown)
            if close[i] < s3_aligned[i] or close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or touches R4 (strong resistance/breakout)
            if close[i] > r3_aligned[i] or close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price bounces from S3 with volume confirmation
            if low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price rejects from R3 with volume confirmation
            elif high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
            # Breakout entries: price breaks through R4/S4 with volume
            elif high[i] > r4_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            elif low[i] < s4_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 09:10
