# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.440 | +50.7% | -12.9% | 100 | PASS |
| ETHUSDT | 0.058 | +19.5% | -20.3% | 110 | PASS |
| SOLUSDT | 0.750 | +142.3% | -32.6% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.417 | -0.2% | -7.6% | 44 | FAIL |
| ETHUSDT | 0.816 | +24.0% | -9.9% | 32 | PASS |
| SOLUSDT | 0.623 | +20.5% | -10.4% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Daily EMA34 trend filter + Camarilla R1/S1 breakouts with volume confirmation
# reduces false signals while maintaining trend alignment. Works in both bull and bear markets
# by only trading in direction of daily trend. Target: 20-30 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r1 = close_1d + (high_1d - low_1d) * 1.12 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.12 / 12
    
    # Calculate daily volume average (20-period) for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate volume spike on 4h timeframe
    vol_ma_20_4h_calc = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20_4h_calc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with uptrend (above daily EMA34) and volume
            if close[i] > r1_4h[i] and close[i] > ema_34_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with downtrend (below daily EMA34) and volume
            elif close[i] < s1_4h[i] and close[i] < ema_34_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below daily EMA34 (trend change)
            if close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above daily EMA34 (trend change)
            if close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 00:11
