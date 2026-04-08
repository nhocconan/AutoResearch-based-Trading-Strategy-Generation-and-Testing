# Strategy: 6h_camarilla_pivot_daily_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.121 | +25.7% | -10.0% | 309 | PASS |
| ETHUSDT | 0.558 | +58.6% | -12.1% | 293 | PASS |
| SOLUSDT | 1.150 | +204.0% | -19.2% | 246 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.708 | -2.1% | -8.6% | 109 | FAIL |
| ETHUSDT | 0.395 | +12.3% | -10.7% | 110 | PASS |
| SOLUSDT | 0.472 | +14.1% | -10.0% | 96 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h Camarilla Pivot + Daily Trend + Volume Confirmation v1
Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance.
Price bouncing off S3/R3 levels (mean reversion) or breaking through S4/R4 levels (breakout)
with daily trend alignment and volume confirmation works in both bull and bear markets.
Designed for 6h timeframe to capture fewer, higher-quality trades.
Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_daily_trend_volume_v1"
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
    
    # Daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    # Based on previous day's OHLC
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + range_hl * 1.1 / 2
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    s4 = prev_close - range_hl * 1.1 / 2
    
    # Daily EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Align to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4.values)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4.values)
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21.values)
    
    # Volume filter (>1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_21_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or trend reverses
            if close[i] <= s3_6h[i] or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or trend reverses
            if close[i] >= r3_6h[i] or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at S3 with trend alignment
            if (close[i] <= s3_6h[i] and 
                close[i] > ema_21_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout long at R4 with trend alignment
            elif (close[i] >= r4_6h[i] and 
                  close[i] > ema_21_6h[i] and 
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at R3 with trend alignment
            elif (close[i] >= r3_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Breakout short at S4 with trend alignment
            elif (close[i] <= s4_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 00:20
