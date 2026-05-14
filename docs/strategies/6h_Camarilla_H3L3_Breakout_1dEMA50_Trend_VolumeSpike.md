# Strategy: 6h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.173 | +26.9% | -4.8% | 240 | KEEP |
| ETHUSDT | 0.228 | +30.0% | -12.4% | 204 | KEEP |
| SOLUSDT | 0.496 | +57.4% | -13.4% | 164 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.411 | +2.9% | -4.2% | 82 | DISCARD |
| ETHUSDT | 1.449 | +25.3% | -5.5% | 77 | KEEP |
| SOLUSDT | 0.339 | +9.9% | -6.3% | 59 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
6h Camarilla Pivot H3L3 Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as magnet points on 1d timeframe.
Breakouts above H3 (resistance) or below L3 (support) with volume confirmation
and aligned with 1d EMA50 trend capture momentum in both bull and bear markets.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Designed for 6h timeframe with tight entry conditions to achieve 12-37 trades/year.
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
    
    # Get 1d data for Camarilla pivot and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d
    # Based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    h3 = pivot + range_val * 1.1 / 4.0  # Resistance level 3
    l3 = pivot - range_val * 1.1 / 4.0  # Support level 3
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate EMA50 on 1d close for trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 (resistance) AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 (support) AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < l3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below H3 OR price crosses below EMA (trend change)
            if (curr_low < h3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above L3 OR price crosses above EMA (trend change)
            if (curr_high > l3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 06:04
