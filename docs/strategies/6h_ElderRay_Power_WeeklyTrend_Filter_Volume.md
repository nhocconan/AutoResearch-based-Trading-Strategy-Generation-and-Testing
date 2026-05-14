# Strategy: 6h_ElderRay_Power_WeeklyTrend_Filter_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.219 | +14.9% | -7.4% | 248 | FAIL |
| ETHUSDT | 0.217 | +28.6% | -7.2% | 189 | PASS |
| SOLUSDT | 0.815 | +77.5% | -10.2% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.657 | +12.8% | -4.6% | 77 | PASS |
| SOLUSDT | -1.193 | -5.8% | -13.3% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Elder Ray Power + Weekly Trend Filter + Volume Confirmation
Uses Elder Ray (bull/bear power) from 6h data combined with weekly EMA trend filter.
Long when bull power > 0 and price above weekly EMA, short when bear power < 0 and price below weekly EMA.
Volume confirmation ensures institutional participation.
Designed for low trade frequency with clear trend-following edge in both bull and bear markets.
"""

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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Elder Ray components (13-period EMA for power calculation)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_weekly_ema = price > ema_34_1w_aligned[i]
        below_weekly_ema = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: bull power positive, price above weekly EMA, volume spike
            if (bull_power[i] > 0 and above_weekly_ema and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bear power negative, price below weekly EMA, volume spike
            elif (bear_power[i] < 0 and below_weekly_ema and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: bear power turns negative (trend weakening)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: bull power turns positive (trend weakening)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Power_WeeklyTrend_Filter_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 00:42
