# Strategy: 4h_Camarilla_S3R3_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.294 | +32.1% | -8.7% | 297 | PASS |
| ETHUSDT | 0.630 | +53.1% | -6.9% | 287 | PASS |
| SOLUSDT | 0.280 | +38.6% | -22.5% | 276 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.246 | -3.7% | -7.3% | 109 | FAIL |
| ETHUSDT | 0.754 | +15.8% | -8.0% | 101 | PASS |
| SOLUSDT | 0.899 | +18.1% | -8.4% | 93 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level S3/R3 breakout with volume confirmation and 1d trend filter.
# Uses Camarilla pivot levels from daily data for precise entry points, confirmed by volume spikes and 1d EMA trend.
# Designed to work in both bull and bear markets by following the 1d trend direction.
# Target: 20-50 trades/year per symbol to avoid excessive fee drag.
name = "4h_Camarilla_S3R3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    pivot = (high_prev + low_prev + close_prev) / 3
    range_ = high_prev - low_prev
    S3 = close_prev - 1.1 * range_ / 2
    R3 = close_prev + 1.1 * range_ / 2
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for EMA and pivot calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume spike in uptrend
            long_condition = (close[i] > R3_aligned[i]) and vol_spike[i] and uptrend
            # Short breakdown: price breaks below S3 with volume spike in downtrend
            short_condition = (close[i] < S3_aligned[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R3 or trend turns down
            if (close[i] < R3_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S3 or trend turns up
            if (close[i] > S3_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 22:08
