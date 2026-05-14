# Strategy: 4h_Camarilla_R1_S1_12hEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.131 | +25.6% | -7.9% | 297 | PASS |
| ETHUSDT | 0.163 | +27.4% | -8.6% | 279 | PASS |
| SOLUSDT | 0.195 | +31.0% | -22.8% | 236 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.038 | -2.0% | -7.0% | 115 | FAIL |
| ETHUSDT | 0.727 | +14.7% | -7.8% | 103 | PASS |
| SOLUSDT | 0.661 | +13.9% | -8.4% | 84 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 12h HTF for EMA34 trend alignment.
- Camarilla pivot levels calculated from prior 1d high/low/close.
- Breakout logic: long when price closes above R1 with volume spike and uptrend,
                  short when price closes below S1 with volume spike and downtrend.
- Trend filter: only long when 4h close > 12h EMA34, only short when 4h close < 12h EMA34.
- Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA (strict to reduce trades).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from prior 1d OHLC (using mtf_data for 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior 1d OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align 1d data to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Camarilla calculations (based on prior 1d range)
    camarilla_range = high_1d_aligned - low_1d_aligned
    camarilla_R1 = close_1d_aligned + camarilla_range * 1.1 / 12
    camarilla_S1 = close_1d_aligned - camarilla_range * 1.1 / 12
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (strict)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 4h close vs 12h EMA34
    uptrend = close > ema_34_12h_aligned
    downtrend = close < ema_34_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Need 12h EMA34 and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_R1[i]) or 
            np.isnan(camarilla_S1[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above R1 AND uptrend AND volume spike
            if close[i] > camarilla_R1[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below S1 AND downtrend AND volume spike
            elif close[i] < camarilla_S1[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint of Camarilla levels or reverse signal
            camarilla_mid = (camarilla_R1[i] + camarilla_S1[i]) / 2
            if close[i] <= camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint of Camarilla levels or reverse signal
            camarilla_mid = (camarilla_R1[i] + camarilla_S1[i]) / 2
            if close[i] >= camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 06:43
