# Strategy: 4h_ChaikinBreakout_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.095 | +24.4% | -13.6% | 295 | PASS |
| ETHUSDT | 0.564 | +53.7% | -9.0% | 271 | PASS |
| SOLUSDT | 0.711 | +91.0% | -24.9% | 258 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.998 | -3.1% | -8.9% | 99 | FAIL |
| ETHUSDT | 1.008 | +21.7% | -7.2% | 99 | PASS |
| SOLUSDT | 0.109 | +7.0% | -12.1% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_ChaikinBreakout_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Chaikin Oscillator (3,10) - momentum indicator
    # ADL = cumulative ((Close - Low) - (High - Close)) / (High - Low) * Volume
    adl = np.zeros(n)
    for i in range(n):
        if high[i] == low[i]:
            adl[i] = adl[i-1] if i > 0 else 0
        else:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            adl[i] = (adl[i-1] if i > 0 else 0) + clv * volume[i]
    
    # Chaikin Oscillator = EMA(3,ADL) - EMA(10,ADL)
    def ema(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema3 = ema(adl, 3)
    ema10 = ema(adl, 10)
    chaikin = ema3 - ema10
    
    # 2. 12h trend filter: EMA(50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 3. Volume filter: current volume > 1.8 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(chaikin[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        chaikin_positive = chaikin[i] > 0
        chaikin_negative = chaikin[i] < 0
        price_above_ema = close[i] > ema50_12h_aligned[i]
        price_below_ema = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # LONG: Chaikin positive + price above 12h EMA + volume confirmation
            if chaikin_positive and price_above_ema and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin negative + price below 12h EMA + volume confirmation
            elif chaikin_negative and price_below_ema and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin turns negative or price below 12h EMA
            if chaikin_negative or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin turns positive or price above 12h EMA
            if chaikin_positive or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 11:29
