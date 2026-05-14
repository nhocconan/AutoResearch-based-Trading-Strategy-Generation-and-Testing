# Strategy: 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.021 | +20.1% | -5.3% | 70 | FAIL |
| ETHUSDT | 0.225 | +30.2% | -7.3% | 63 | PASS |
| SOLUSDT | 0.107 | +24.2% | -26.4% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.479 | +12.3% | -5.9% | 24 | PASS |
| SOLUSDT | -1.442 | -11.6% | -17.1% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE
# Hypothesis: On 12h timeframe, use daily Camarilla R3/S3 levels as strong support/resistance.
# Enter long when price breaks above R3 with volume spike and daily uptrend (close > EMA34).
# Enter short when price breaks below S3 with volume spike and daily downtrend (close < EMA34).
# Exit when price returns to the opposite level (S3 for longs, R3 for shorts).
# Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Uses 1d trend filter and volume confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets via trend filter.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE"
timeframe = "12h"
leverage = 1.0

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
    
    # Camarilla levels from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R3, S3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA for trend filter (34-period)
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and daily uptrend
            if close[i] > R3_aligned[i] and vol_spike[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and daily downtrend
            elif close[i] < S3_aligned[i] and vol_spike[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S3 level
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R3 level
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 09:37
