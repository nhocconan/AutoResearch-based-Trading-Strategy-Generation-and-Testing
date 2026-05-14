# Strategy: 4h_Pivot_R1_S1_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.143 | +26.5% | -11.9% | 363 | PASS |
| ETHUSDT | 0.163 | +28.0% | -11.3% | 331 | PASS |
| SOLUSDT | 0.609 | +73.2% | -24.0% | 272 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.797 | -8.4% | -13.3% | 146 | FAIL |
| ETHUSDT | 0.660 | +15.1% | -5.6% | 122 | PASS |
| SOLUSDT | 0.385 | +11.0% | -10.2% | 96 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels from 1d with volume confirmation and trend filter on 4h timeframe.
Long when price breaks above S1 with volume spike and 4h price above 20 EMA (uptrend).
Short when price breaks below R1 with volume spike and 4h price below 20 EMA (downtrend).
Uses volume spike (>2x 20-period average) and EMA20 trend filter to avoid false breakouts.
Target: 20-30 trades/year to minimize fee drag while capturing institutional breakout moves.
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
    
    # Daily Camarilla pivot levels (calculated from previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 based on previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align to 4h timeframe (use previous day's levels for current day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # EMA20 for trend filter on 4h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema20 = ema_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above S1 with volume spike in uptrend
            if (price > s1_level and          # breaks above S1
                vol_spike and                 # volume confirmation
                price > ema20):               # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1 with volume spike in downtrend
            elif (price < r1_level and        # breaks below R1
                  vol_spike and               # volume confirmation
                  price < ema20):             # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below S1 or trend reverses
            if price < s1_level or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above R1 or trend reverses
            if price > r1_level or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 03:41
