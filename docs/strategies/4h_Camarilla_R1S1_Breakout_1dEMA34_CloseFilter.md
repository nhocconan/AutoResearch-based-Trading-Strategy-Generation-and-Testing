# Strategy: 4h_Camarilla_R1S1_Breakout_1dEMA34_CloseFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.309 | +38.2% | -12.2% | 56 | PASS |
| ETHUSDT | 0.107 | +24.4% | -17.9% | 59 | PASS |
| SOLUSDT | 0.743 | +126.3% | -30.6% | 50 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.560 | -1.2% | -7.6% | 24 | FAIL |
| ETHUSDT | 0.996 | +27.8% | -7.3% | 17 | PASS |
| SOLUSDT | -0.334 | -2.5% | -18.6% | 16 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dEMA34_CloseFilter
# Hypothesis: On 4h timeframe, enter long when price closes above daily R1 with close > daily EMA34 and volume > 2x average.
# Enter short when price closes below daily S1 with close < daily EMA34 and volume > 2x average.
# Exit when price crosses daily EMA34 (trend reversal).
# Uses close-based filters only to avoid look-ahead. Targets 20-30 trades/year for low fee drag.
# Works in bull markets via breakouts and in bear via short reversals at S1.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_CloseFilter"
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
    
    # Load daily data for Camarilla pivot calculation and EMA34
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
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1d_trend = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R1 with volume > 2x average and close > daily EMA34
            if close[i] > r1_val and close[i] > ema1d_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with volume > 2x average and close < daily EMA34
            elif close[i] < s1_val and close[i] < ema1d_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily EMA34 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily EMA34 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 08:41
