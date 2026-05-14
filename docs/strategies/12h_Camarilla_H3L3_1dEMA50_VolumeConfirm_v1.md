# Strategy: 12h_Camarilla_H3L3_1dEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.451 | +9.3% | -7.7% | 72 | DISCARD |
| ETHUSDT | 0.001 | +20.9% | -6.5% | 63 | KEEP |
| SOLUSDT | 0.142 | +27.1% | -24.1% | 61 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.251 | +8.4% | -3.1% | 28 | KEEP |
| SOLUSDT | -0.724 | -1.7% | -10.5% | 23 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment (proven pattern from DB)
- Camarilla pivot levels (H3, L3) calculated from previous completed 1d bar
- Long when price breaks above H3 AND price > 1d EMA50 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below L3 AND price < 1d EMA50 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the Camarilla pivot point (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 12h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (H3, L3, pivot)
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla formula:
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Pivot = (high + low + close) / 3
    camarilla_h3 = close_1d_prev + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3 = close_1d_prev - 1.1 * (high_1d - low_1d) / 4.0
    camarilla_pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at open)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla pivot point
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla pivot point
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-24 05:50
