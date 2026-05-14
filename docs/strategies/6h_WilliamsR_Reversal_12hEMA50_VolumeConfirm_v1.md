# Strategy: 6h_WilliamsR_Reversal_12hEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.621 | +4.4% | -14.0% | 113 | FAIL |
| ETHUSDT | 0.507 | +42.5% | -13.8% | 93 | PASS |
| SOLUSDT | 0.382 | +42.4% | -13.7% | 74 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.068 | +20.6% | -4.8% | 41 | PASS |
| SOLUSDT | -1.459 | -9.0% | -11.8% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
- Williams %R identifies overbought/oversold conditions; reversal signals when %R crosses above/below -50.
- Trend filter: price must be above/below 12h EMA50 to align with higher timeframe direction.
- Volume confirmation: current volume > 1.5x 20-bar average to ensure conviction.
- Designed for 6h timeframe to capture medium-term reversals in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-25 trades/year (50-100 total over 4 years) to stay fee-efficient.
"""

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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h close for EMA (completed 12h bar)
    close_12h = df_12h['close'].shift(1).values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Avoid division by zero in Williams %R
        if highest_high[i] == lowest_low[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -50 (oversold reversal) AND price above 12h EMA50 AND volume confirmation
            if williams_r[i] > -50 and williams_r[i-1] <= -50 and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50 (overbought reversal) AND price below 12h EMA50 AND volume confirmation
            elif williams_r[i] < -50 and williams_r[i-1] >= -50 and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR price crosses below 12h EMA50
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR price crosses above 12h EMA50
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 00:33
