# Strategy: 4H_ADX_WilliamsR_Trend_Momentum

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.902 | -14.6% | -27.9% | 150 | FAIL |
| ETHUSDT | 0.145 | +27.2% | -15.4% | 152 | PASS |
| SOLUSDT | 0.093 | +22.9% | -24.5% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.544 | +13.3% | -12.0% | 56 | PASS |
| SOLUSDT | -0.093 | +3.5% | -15.0% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4-hour ADX trend strength combined with 12-hour Williams %R momentum.
Long when ADX > 25 (trending) and Williams %R crosses above -50 (bullish momentum).
Short when ADX > 25 (trending) and Williams %R crosses below -50 (bearish momentum).
Exit when ADX falls below 20 (trend weakening) or Williams %R crosses back through -50.
Designed for moderate trade frequency (~30-50/year) to capture trends while minimizing whipsaws in ranging markets.
"""

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
    
    # Load 12-hour data for ADX and Williams %R - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Calculate 12-hour Williams %R (14-period)
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r_values = williams_r.values
    
    # Align HTF indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        williams_r_val = williams_r_aligned[i]
        
        if position == 0:
            # Long: ADX > 25 (strong trend) and Williams %R crosses above -50 (bullish momentum)
            if adx_val > 25 and williams_r_val > -50 and williams_r_aligned[i-1] <= -50:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) and Williams %R crosses below -50 (bearish momentum)
            elif adx_val > 25 and williams_r_val < -50 and williams_r_aligned[i-1] >= -50:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (weakening trend) or Williams %R crosses back below -50
                if adx_val < 20 or (williams_r_val < -50 and williams_r_aligned[i-1] >= -50):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (weakening trend) or Williams %R crosses back above -50
                if adx_val < 20 or (williams_r_val > -50 and williams_r_aligned[i-1] <= -50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_ADX_WilliamsR_Trend_Momentum"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 00:16
