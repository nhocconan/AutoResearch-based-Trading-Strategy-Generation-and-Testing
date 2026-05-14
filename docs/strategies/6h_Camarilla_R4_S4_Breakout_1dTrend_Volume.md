# Strategy: 6h_Camarilla_R4_S4_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.268 | +27.7% | -5.6% | 81 | PASS |
| ETHUSDT | 0.087 | +23.7% | -6.0% | 62 | PASS |
| SOLUSDT | -0.390 | +2.2% | -12.9% | 65 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.857 | +0.8% | -4.0% | 34 | FAIL |
| ETHUSDT | 0.783 | +13.7% | -5.3% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_Camarilla_R4_S4_Breakout_1dTrend_Volume
# Hypothesis: Use daily Camarilla pivot levels (R4/S4) for breakout entries on 6h timeframe.
# Enter long on break above R4 with volume confirmation and daily trend filter.
# Enter short on break below S4 with volume confirmation and daily trend filter.
# Exit when price returns to daily pivot (R3/S3) or trend reverses.
# This strategy captures strong momentum breaks while filtering with higher timeframe trend.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Targets 15-25 trades/year by requiring confluence of level break, volume, and trend.

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate daily Camarilla levels
    # Based on previous day's high, low, close
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate pivot and ranges
    pivot = (ph + pl + pc) / 3
    range_val = ph - pl
    
    # Camarilla levels
    r4 = pc + range_val * 1.1 / 2
    r3 = pc + range_val * 1.1 / 4
    s3 = pc - range_val * 1.1 / 4
    s4 = pc - range_val * 1.1 / 2
    
    # Align daily levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2x average of last 4 periods (1 day)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from daily EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]

        if position == 0:
            # LONG: break above R4 with volume and uptrend
            if close[i] > r4_aligned[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S4 with volume and downtrend
            elif close[i] < s4_aligned[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to R3 or trend turns down
            if close[i] < r3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to S3 or trend turns up
            if close[i] > s3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 15:34
