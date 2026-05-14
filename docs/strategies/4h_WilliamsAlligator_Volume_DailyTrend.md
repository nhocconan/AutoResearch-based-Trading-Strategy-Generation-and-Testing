# Strategy: 4h_WilliamsAlligator_Volume_DailyTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.276 | +34.3% | -11.6% | 159 | PASS |
| ETHUSDT | 0.209 | +31.9% | -13.4% | 160 | PASS |
| SOLUSDT | 0.858 | +134.6% | -19.0% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.303 | -7.7% | -9.9% | 62 | FAIL |
| ETHUSDT | 0.611 | +16.2% | -7.9% | 48 | PASS |
| SOLUSDT | -0.571 | -4.5% | -15.5% | 45 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h Williams Alligator + Volume Spike + Daily Trend
# Hypothesis: Williams Alligator identifies trend formation (jaws/teeth/lips divergence).
# Jaw (13-period smoothed), Teeth (8-period), Lips (5-period). 
# Strong uptrend when Lips > Teeth > Jaws; strong downtrend when Lips < Teeth < Jaws.
# Combined with volume spikes for confirmation and daily EMA trend filter to align with higher timeframe trend.
# Works in both bull and bear markets by following Alligator-defined trends.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "4h_WilliamsAlligator_Volume_DailyTrend"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Williams Alligator (periods: 13, 8, 5) ===
    # Smoothed Moving Average (SMMA) function
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA formula
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(high, 13)  # Jaw (Blue) - 13-period SMMA of Median Price
    teeth = smma(high, 8)  # Teeth (Red) - 8-period SMMA of Median Price
    lips = smma(high, 5)   # Lips (Green) - 5-period SMMA of Median Price
    
    # Use median price (typical price) for Alligator calculation
    median_price = (high + low) / 2
    jaws = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaws (bullish alignment) + volume spike + price above daily EMA34
            if (lips[i] > teeth[i] and teeth[i] > jaws[i] and 
                vol_spike[i] and
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (bearish alignment) + volume spike + price below daily EMA34
            elif (lips[i] < teeth[i] and teeth[i] < jaws[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator lines cross (Lips < Teeth or Teeth < Jaws) - trend weakening
            if lips[i] < teeth[i] or teeth[i] < jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator lines cross (Lips > Teeth or Teeth > Jaws) - trend weakening
            if lips[i] > teeth[i] or teeth[i] > jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 06:46
