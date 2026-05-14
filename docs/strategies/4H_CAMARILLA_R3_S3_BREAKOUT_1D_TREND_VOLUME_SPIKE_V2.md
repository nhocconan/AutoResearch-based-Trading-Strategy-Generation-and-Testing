# Strategy: 4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE_V2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.068 | +23.2% | -10.8% | 101 | PASS |
| ETHUSDT | 0.367 | +40.4% | -8.8% | 87 | PASS |
| SOLUSDT | 0.935 | +126.3% | -15.2% | 74 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.080 | -3.9% | -9.9% | 36 | FAIL |
| ETHUSDT | 1.414 | +29.8% | -6.1% | 31 | PASS |
| SOLUSDT | -0.453 | -0.7% | -11.4% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE_V2
# Hypothesis: Optimize the original strategy by tightening volume confirmation and adding a momentum filter.
# Uses tighter volume threshold (2.5x average) and requires price to be above/below 5-period EMA for momentum confirmation.
# Aims to reduce trade frequency while maintaining edge in both bull and bear markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE_V2"
timeframe = "4h"
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
    
    # Camarilla formulas: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA for trend filter (34-period)
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume spike detection (20-period volume MA) - tightened threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.5  # Increased from 2.0 to 2.5 for fewer trades
    
    # Momentum confirmation: 5-period EMA on 4h close
    ema5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(ema5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike, daily uptrend, and bullish momentum
            if (close[i] > R3_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i] and close[i] > ema5[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike, daily downtrend, and bearish momentum
            elif (close[i] < S3_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i] and close[i] < ema5[i]):
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
2026-05-12 09:42
