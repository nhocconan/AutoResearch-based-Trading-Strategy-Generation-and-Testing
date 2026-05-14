# Strategy: 4h_12hCamarilla_R3_S3_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.367 | +7.4% | -15.2% | 429 | FAIL |
| ETHUSDT | 0.052 | +22.3% | -8.6% | 409 | PASS |
| SOLUSDT | -0.052 | +13.5% | -18.3% | 333 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.289 | +9.5% | -7.0% | 140 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation
# Long when price breaks above 12h Camarilla R3 level AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 level AND volume > 1.5 * avg_volume(20)
# Exit when price returns to 12h Camarilla midpoint (PP) or opposite Camarilla level touched
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla R3/S3 represent strong breakout levels from 12h structure
# Volume filter ensures institutional participation, reducing false breakouts
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "4h_12hCamarilla_R3_S3_Breakout_Volume"
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
    
    # Get 12h data ONCE before loop for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for Camarilla
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 4.0
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 4.0
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3 level with volume confirmation
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3 level with volume confirmation
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h Camarilla PP or touches S3 (reversal)
            if close[i] <= pp_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h Camarilla PP or touches R3 (reversal)
            if close[i] >= pp_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 05:39
