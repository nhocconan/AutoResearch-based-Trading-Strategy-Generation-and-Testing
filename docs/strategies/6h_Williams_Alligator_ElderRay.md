# Strategy: 6h_Williams_Alligator_ElderRay

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.617 | +43.6% | -4.9% | 171 | PASS |
| ETHUSDT | 0.347 | +36.2% | -8.8% | 172 | PASS |
| SOLUSDT | 1.768 | +286.7% | -11.5% | 160 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.390 | -1.6% | -4.6% | 50 | FAIL |
| ETHUSDT | 0.938 | +16.6% | -7.2% | 45 | PASS |
| SOLUSDT | -0.742 | -1.0% | -6.2% | 49 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Williams_Alligator_ElderRay
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction, while Elder Ray (Bull/Bear Power) confirms momentum strength. 
Long when price > Teeth, Bull Power > 0, and Bear Power < 0. Short when price < Teeth, Bull Power < 0, and Bear Power > 0.
Uses 12h trend filter for higher timeframe bias. Works in trending markets (both bull/bear) and avoids chop via Alligator alignment.
Target: 15-35 trades/year per symbol.
"""

name = "6h_Williams_Alligator_ElderRay"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Alligator alignment: check if aligned (all in order)
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        bull = bull_power[i]
        bear = bear_power[i]
        
        uptrend_htf = uptrend_12h_aligned[i]
        downtrend_htf = downtrend_12h_aligned[i]
        
        if position == 0:
            # LONG: price > Teeth, Bull Power > 0, Bear Power < 0, 12h uptrend
            if close[i] > teeth_val and bull > 0 and bear < 0 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Teeth, Bull Power < 0, Bear Power > 0, 12h downtrend
            elif close[i] < teeth_val and bull < 0 and bear > 0 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Teeth or Bull Power <= 0
            if close[i] < teeth_val or bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Teeth or Bear Power <= 0
            if close[i] > teeth_val or bear <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 07:04
