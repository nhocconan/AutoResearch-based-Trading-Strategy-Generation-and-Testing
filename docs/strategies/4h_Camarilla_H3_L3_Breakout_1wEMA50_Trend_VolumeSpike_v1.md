# Strategy: 4h_Camarilla_H3_L3_Breakout_1wEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.691 | +35.7% | -2.9% | 202 | PASS |
| ETHUSDT | 0.478 | +33.3% | -4.5% | 172 | PASS |
| SOLUSDT | -0.176 | +14.3% | -13.5% | 125 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.781 | -5.5% | -6.0% | 79 | FAIL |
| ETHUSDT | 0.143 | +7.1% | -5.1% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels represent stronger intraday support/resistance than R1/S1.
Breakouts above H3 or below L3 with volume confirmation and weekly EMA50 trend filter capture
strong institutional moves. Weekly trend filter ensures alignment with higher timeframe momentum,
working in both bull/bear markets by only taking breakouts in direction of weekly trend.
Target: 20-50 trades/year (75-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3.0
    
    # Shift to get previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar: use current values (will be filtered out by min_periods anyway)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels (H3/L3 are stronger than R1/S1)
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla calculation (1) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        h3_level = H3[i]
        l3_level = L3[i]
        pivot_level = pivot[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above H3 AND above weekly EMA50 (uptrend filter)
            long_condition = (curr_close > h3_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below L3 AND below weekly EMA50 (downtrend filter)
            short_condition = (curr_close < l3_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot or trend breaks
            if curr_close <= pivot_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot or trend breaks
            if curr_close >= pivot_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 01:16
