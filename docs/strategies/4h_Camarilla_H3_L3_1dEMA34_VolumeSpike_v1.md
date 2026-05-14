# Strategy: 4h_Camarilla_H3_L3_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.323 | +31.9% | -5.1% | 238 | PASS |
| ETHUSDT | 0.134 | +25.9% | -7.7% | 227 | PASS |
| SOLUSDT | 0.444 | +49.7% | -14.4% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.351 | -3.5% | -7.0% | 94 | FAIL |
| ETHUSDT | 0.918 | +16.3% | -7.6% | 77 | PASS |
| SOLUSDT | 0.098 | +6.9% | -6.1% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for EMA34 trend alignment (proven BTC/ETH edge from DB).
- Camarilla pivot levels calculated from prior 1d high/low/close.
- Breakout logic: long when price closes above H3 with volume spike and uptrend,
                  short when price closes below L3 with volume spike and downtrend.
- Trend filter: only long when 4h close > 1d EMA34, only short when 4h close < 1d EMA34.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d OHLC (using mtf_data for 1d)
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
    camarilla_H3 = close_1d_aligned + camarilla_range * 1.1 / 4
    camarilla_L3 = close_1d_aligned - camarilla_range * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (strict)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 4h close vs 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Need 1d EMA34 and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > camarilla_H3[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < camarilla_L3[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint of Camarilla levels or reverse signal
            camarilla_mid = (camarilla_H3[i] + camarilla_L3[i]) / 2
            if close[i] <= camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint of Camarilla levels or reverse signal
            camarilla_mid = (camarilla_H3[i] + camarilla_L3[i]) / 2
            if close[i] >= camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 06:50
