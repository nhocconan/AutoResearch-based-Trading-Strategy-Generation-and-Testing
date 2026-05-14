# Strategy: 4h_Camarilla_R1S1_Breakout_12hEMA50_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.265 | +35.8% | -15.5% | 94 | PASS |
| ETHUSDT | 0.095 | +23.0% | -15.5% | 100 | PASS |
| SOLUSDT | 1.217 | +284.4% | -22.9% | 82 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.186 | -8.2% | -12.5% | 39 | FAIL |
| ETHUSDT | 0.697 | +20.4% | -8.2% | 31 | PASS |
| SOLUSDT | 0.251 | +10.1% | -11.3% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_12hEMA50_TrendFilter
# Hypothesis: On 4h timeframe, enter long when price closes above daily R1 with close > 12h EMA50.
# Enter short when price closes below daily S1 with close < 12h EMA50.
# Exit when price crosses 12h EMA50 (trend reversal).
# Uses 12h EMA for trend filter to reduce whipsaw and improve performance in both bull and bear markets.
# Targets 20-30 trades/year for low fee drag.
# 12h EMA provides smoother trend signal than daily EMA, reducing false signals during choppy periods.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R1 and S1 levels
    r1 = daily_pivot + daily_range * 1.083
    s1 = daily_pivot - daily_range * 1.083
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily levels and 12h EMA50 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema12h_trend = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R1 with close > 12h EMA50 and volume > 20MA
            if close[i] > r1_val and close[i] > ema12h_trend and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with close < 12h EMA50 and volume > 20MA
            elif close[i] < s1_val and close[i] < ema12h_trend and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 12h EMA50 (trend reversal)
            if close[i] < ema12h_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 12h EMA50 (trend reversal)
            if close[i] > ema12h_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 08:48
