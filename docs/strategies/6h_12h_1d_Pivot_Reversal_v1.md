# Strategy: 6h_12h_1d_Pivot_Reversal_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.200 | +30.3% | -10.0% | 167 | PASS |
| ETHUSDT | 0.133 | +26.5% | -15.5% | 150 | PASS |
| SOLUSDT | 1.078 | +187.1% | -23.0% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.915 | -14.5% | -16.4% | 72 | FAIL |
| ETHUSDT | 1.515 | +36.9% | -8.4% | 45 | PASS |
| SOLUSDT | 0.226 | +9.2% | -18.7% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_12h_1d_Pivot_Reversal_v1
Hypothesis: On 6h timeframe, look for reversals at daily pivot levels (R1/S1, R2/S2) with 12h trend filter.
Long when price breaks above R1 with 12h uptrend; short when breaks below S1 with 12h downtrend.
Exit at opposite pivot level (S1 for longs, R1 for shorts). Uses volume confirmation to avoid false breakouts.
Designed for low trade frequency (15-30/year) by requiring pivot-level confluence and trend alignment.
Works in bull/bear via 12h trend filter and mean-reversion exit at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Pivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY PIVOT LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Support and resistance levels
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + range_1d
    s2_1d = pivot_1d - range_1d
    
    # === 12H EMA(25) FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 25:
        ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    else:
        ema_25_12h = np.full_like(close_12h, np.nan)
    
    # Align data to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Volume average (24-period for 6h = ~6 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema_25_12h_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.8x average
        vol_confirm = volume[i] > 1.8 * vol_avg[i]
        
        # Trend filter: price above/below 12h EMA(25)
        price_above_ema = close[i] > ema_25_12h_aligned[i]
        price_below_ema = close[i] < ema_25_12h_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > r1_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s1_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to opposite pivot level
        exit_long = close[i] < s1_1d_aligned[i]
        exit_short = close[i] > r1_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-12 04:52
