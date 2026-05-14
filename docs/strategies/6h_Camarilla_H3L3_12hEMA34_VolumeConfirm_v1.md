# Strategy: 6h_Camarilla_H3L3_12hEMA34_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.200 | +27.1% | -5.7% | 232 | PASS |
| ETHUSDT | 0.398 | +35.1% | -6.2% | 188 | PASS |
| SOLUSDT | 0.421 | +46.9% | -10.4% | 159 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.040 | -6.9% | -8.5% | 94 | FAIL |
| ETHUSDT | 1.280 | +20.8% | -4.0% | 73 | PASS |
| SOLUSDT | 0.524 | +11.3% | -4.6% | 61 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 12h HTF for EMA34 trend alignment (proven pattern from DB)
- Camarilla levels (H3/L3) calculated from previous completed 12h bar (range of prior 12h candle)
- Long when price breaks above H3 AND price > 12h EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below L3 AND price < 12h EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the Camarilla H4/L4 midpoint (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 6h timeframe recommendation
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
    
    # Calculate 12h EMA34 for trend filter (using previous completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h Camarilla levels (H3/L3) from previous completed 12h bar
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    rango_12h = high_12h - low_12h
    camarilla_h3 = close_12h + 1.1 * rango_12h / 4.0
    camarilla_l3 = close_12h - 1.1 * rango_12h / 4.0
    # Midpoint between H4 and L4 for exit (H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2)
    camarilla_h4 = close_12h + 1.1 * rango_12h / 2.0
    camarilla_l4 = close_12h - 1.1 * rango_12h / 2.0
    camarilla_mid = (camarilla_h4 + camarilla_l4) / 2.0  # Equivalent to close_12h
    
    # Align Camarilla levels to 12h timeframe (previous completed 12h bar values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 12h EMA34
    uptrend = close > ema_34_12h_aligned
    downtrend = close < ema_34_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 12h EMA34, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
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
            # Long exit: price reverts to Camarilla midpoint (close_12h)
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla midpoint (close_12h)
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_12hEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 05:52
