# Strategy: 6H_Pivot_R1_S1_Breakout_1W_EMA50_Trend_Volume_Spike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.234 | +16.4% | -5.7% | 96 | FAIL |
| ETHUSDT | 0.003 | +21.4% | -6.7% | 79 | PASS |
| SOLUSDT | 0.329 | +36.3% | -11.8% | 53 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.998 | +13.9% | -3.4% | 28 | PASS |
| SOLUSDT | -0.072 | +5.4% | -7.9% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend and daily for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous day's pivot points (standard)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = high_1d
    prev_low = low_1d
    prev_close = close_1d
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + above weekly EMA50 + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + below weekly EMA50 + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above pivot (full exit)
            if position == 1:
                # Exit long: Price closes below pivot
                if close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above pivot
                if close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Pivot_R1_S1_Breakout_1W_EMA50_Trend_Volume_Spike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 11:17
