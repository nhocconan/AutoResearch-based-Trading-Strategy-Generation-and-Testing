# Strategy: 12h_camarilla_pivot_1d_trend_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.439 | +0.3% | -17.2% | 159 | FAIL |
| ETHUSDT | 0.223 | +32.4% | -13.3% | 145 | PASS |
| SOLUSDT | 0.646 | +92.5% | -27.0% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.352 | +11.3% | -11.7% | 50 | PASS |
| SOLUSDT | -0.628 | -5.8% | -19.6% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: Camarilla pivot levels on 1d timeframe identify key support/resistance levels.
Trades when price breaks above/below pivot with volume confirmation and 1d trend alignment.
Works in both bull and bear markets by trading breakouts from established levels.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use R3, R2, S3, S2 levels
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + range_hl * 1.1 / 4
    r2 = pivot + range_hl * 1.1 / 6
    s3 = pivot - range_hl * 1.1 / 4
    s2 = pivot - range_hl * 1.1 / 6
    
    # Daily EMA for trend filter (20-period)
    ema_20 = df_1d['close'].ewm(span=20, adjust=False).mean()
    
    # Align all daily data to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20.values)
    
    # Volume confirmation (10-period average = 5 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S2 or trend turns bearish
            if close[i] < s2_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R2 or trend turns bullish
            if close[i] > r2_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R2 with volume and bullish trend
            if (close[i] > r2_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S2 with volume and bearish trend
            elif (close[i] < s2_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 15:18
