# Strategy: 4h_Camarilla_R1S1_1dEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.119 | +25.2% | -8.6% | 264 | PASS |
| ETHUSDT | 0.111 | +25.1% | -8.4% | 248 | PASS |
| SOLUSDT | 0.649 | +74.6% | -16.8% | 209 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.967 | -1.3% | -4.2% | 94 | FAIL |
| ETHUSDT | 0.924 | +18.7% | -9.2% | 90 | PASS |
| SOLUSDT | 0.767 | +16.1% | -7.2% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout + 1d EMA50 trend filter + volume spike confirmation
- Camarilla pivot levels from 1d provide high-probability intraday support/resistance with proven edge
- Breakout above R1 or below S1 with volume confirmation captures institutional moves
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Discrete position sizing (0.25) minimizes fee churn
- Target: 20-40 trades/year per symbol (~80-160 total over 4 years)
- Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakdowns in downtrend)
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
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from 1d OHLC
    def calculate_camarilla(high_arr, low_arr, close_arr):
        # Typical price
        pp = (high_arr + low_arr + close_arr) / 3.0
        # Range
        rng = high_arr - low_arr
        # Camarilla levels
        r1 = pp + (rng * 1.1 / 12)
        s1 = pp - (rng * 1.1 / 12)
        return r1, s1
    
    camarilla_r1_1d, camarilla_s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R1 + volume spike + price > 1d EMA50 (uptrend)
            if price > r1 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + price < 1d EMA50 (downtrend)
            elif price < s1 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint between R1 and S1
            mid_point = (r1 + s1) / 2.0
            if price < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midpoint between R1 and S1
            mid_point = (r1 + s1) / 2.0
            if price > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:10
