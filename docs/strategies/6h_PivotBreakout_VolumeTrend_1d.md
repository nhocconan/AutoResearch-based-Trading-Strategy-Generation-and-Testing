# Strategy: 6h_PivotBreakout_VolumeTrend_1d

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.024 | +22.1% | -3.5% | 109 | PASS |
| ETHUSDT | 0.169 | +26.5% | -5.3% | 93 | PASS |
| SOLUSDT | -0.145 | +10.2% | -14.2% | 100 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.363 | -7.6% | -8.1% | 47 | FAIL |
| ETHUSDT | 1.221 | +19.0% | -5.4% | 39 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PivotBreakout_VolumeTrend_1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Pivot point and R1/S1 levels (standard calculation)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + range_val
    s1 = pivot - range_val
    
    # Align pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA20 trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1d = (close_1d > ema20_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Price distance filter: require breakout to be at least 0.3% above/below level
    price_above_r1 = close > r1_6h * 1.003
    price_below_s1 = close < s1_6h * 0.997
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and 1d uptrend
            long_cond = (price_above_r1[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below S1 with volume spike and 1d downtrend
            short_cond = (price_below_s1[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R1 (mean reversion)
            if close[i] < r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above S1 (mean reversion)
            if close[i] > s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Pivot R1/S1 breakout with volume confirmation and 1d trend filter on 6h timeframe.
# Uses standard pivot points (not Camarilla) for cleaner breakout signals.
# Volume spike >2x 20-period average ensures institutional participation.
# Price distance filter (0.3%) avoids false breakouts from noise.
# Trend filter ensures alignment with daily bias.
# Target: 20-40 trades/year to minimize fee decay while capturing meaningful moves.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at S1/R1).
```

## Last Updated
2026-05-08 10:37
