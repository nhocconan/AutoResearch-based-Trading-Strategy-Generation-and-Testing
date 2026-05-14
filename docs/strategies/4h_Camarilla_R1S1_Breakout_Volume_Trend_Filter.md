# Strategy: 4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.266 | +28.7% | -7.5% | 343 | PASS |
| ETHUSDT | 0.031 | +22.0% | -10.2% | 313 | PASS |
| SOLUSDT | -0.050 | +16.4% | -16.2% | 269 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.392 | -1.8% | -4.8% | 136 | FAIL |
| ETHUSDT | 1.045 | +17.6% | -4.1% | 123 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter
Hypothesis: Trade 4h Camarilla R1/S1 level breakouts with volume confirmation and 1d trend filter.
Long when price breaks above R1 with volume spike and 1d uptrend; short when breaks below S1 with volume spike and 1d downtrend.
Camarilla levels provide institutional support/resistance, volume filter reduces false breakouts, and 1d trend filter avoids counter-trend trades.
Target: 75-200 total trades over 4 years (19-50/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, volume filter reduces false breakouts.
"""

name = "4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = multiplier * close_1d[i] + (1 - multiplier) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h Camarilla levels (R1, S1)
    camarilla_R1 = np.full_like(high, np.nan)
    camarilla_S1 = np.full_like(low, np.nan)
    for i in range(1, n):
        # Camarilla uses previous period's OHLC
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        range_val = prev_high - prev_low
        camarilla_R1[i] = prev_close + (range_val * 1.1 / 12)
        camarilla_S1[i] = prev_close - (range_val * 1.1 / 12)
    
    # Calculate volume filter (volume > 2.0x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume filter AND 1d uptrend
            if close[i] > camarilla_R1[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume filter AND 1d downtrend
            elif close[i] < camarilla_S1[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 OR 1d trend turns down
            if close[i] < camarilla_S1[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 OR 1d trend turns up
            if close[i] > camarilla_R1[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 04:52
