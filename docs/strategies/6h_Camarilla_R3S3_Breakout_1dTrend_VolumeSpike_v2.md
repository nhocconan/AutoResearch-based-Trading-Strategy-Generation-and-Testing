# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.590 | +47.2% | -6.1% | 153 | PASS |
| ETHUSDT | 0.256 | +32.8% | -12.3% | 143 | PASS |
| SOLUSDT | 0.613 | +75.0% | -14.6% | 125 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.441 | -6.3% | -12.0% | 67 | FAIL |
| ETHUSDT | 1.450 | +29.3% | -6.1% | 49 | PASS |
| SOLUSDT | 0.170 | +7.9% | -7.8% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla pivot levels from daily data provide robust support/resistance. 
Breakouts above R3 or below S3 with volume confirmation and aligned daily trend 
capture institutional moves while minimizing false breakouts. Designed for low 
trade frequency (target: 15-35 trades/year) to minimize fee drift. Works in both 
bull and bear regimes by following the daily trend direction only.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
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
    
    # Get daily data for Camarilla levels and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    hl_range = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + hl_range * 1.1 / 4
    s3 = df_1d['close'] - hl_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R3 with volume spike and above daily EMA34 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with volume spike and below daily EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below S3 or trend turns down
            if (close[i] < s3_aligned[i] or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R3 or trend turns up
            if (close[i] > r3_aligned[i] or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 08:10
