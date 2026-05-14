# Strategy: 4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_VOLUME_SPIKE

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.451 | +42.3% | -8.3% | 203 | PASS |
| ETHUSDT | 0.004 | +19.0% | -13.5% | 200 | PASS |
| SOLUSDT | 0.919 | +130.6% | -20.6% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.380 | -7.4% | -7.8% | 78 | FAIL |
| ETHUSDT | 0.896 | +20.6% | -8.1% | 69 | PASS |
| SOLUSDT | -0.075 | +4.1% | -9.5% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_VOLUME_SPIKE
# Hypothesis: Camarilla R1/S1 breakouts with volume spike and daily EMA trend filter.
# Works in bull/bear: EMA34 filter ensures trades align with higher timeframe trend,
# while Camarilla levels provide institutional-grade support/resistance with built-in risk-reward.
# Volume spike confirms institutional participation. Target: 25-40 trades/year.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_VOLUME_SPIKE"
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
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Typical price for Camarilla calculation
    typical_price = (prev_high + prev_low + prev_close) / 3
    range_h_l = prev_high - prev_low
    
    # Camarilla levels (R1, S1, R2, S2, R3, S3, R4, S4)
    # We focus on R1 and S1 for breakouts
    r1 = typical_price + range_h_l * 1.1 / 12
    s1 = typical_price - range_h_l * 1.1 / 12
    
    # 1-day EMA for trend filter (34-period)
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume spike detection (20-period volume MA on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation in bullish trend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation in bearish trend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S1 (mean reversion to support)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R1 (mean reversion to resistance)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 09:44
