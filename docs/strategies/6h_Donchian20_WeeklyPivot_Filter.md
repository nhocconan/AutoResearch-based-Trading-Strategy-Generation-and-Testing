# Strategy: 6h_Donchian20_WeeklyPivot_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.041 | +19.1% | -11.0% | 110 | FAIL |
| ETHUSDT | 0.036 | +21.6% | -7.4% | 93 | PASS |
| SOLUSDT | -0.088 | +10.8% | -35.2% | 84 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.275 | +8.6% | -5.1% | 21 | PASS |

## Code
```python
# ==========================================================
# Strategy: 6h_Donchian20_WeeklyPivot_Filter
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter.
# - Uses weekly pivot points (from weekly OHLC) to determine trend direction.
# - Long only when price is above weekly pivot; short only when below.
# - Volume confirmation (2x average volume) to filter breakouts.
# - Designed for 6h timeframe: expects ~15-30 trades/year per symbol.
# - Works in bull/bear: pivot adapts to weekly structure; volume avoids false breaks.
# ==========================================================
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3.0
    
    # Align weekly pivot to 6h (no extra delay needed for pivot)
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    
    # Calculate 14-period ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 20-period high/low for Donchian breakout
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 20
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above weekly pivot
            if price > high_max[i] and vol_ratio > 2.0 and price > pivot_w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below weekly pivot
            elif price < low_min[i] and vol_ratio > 2.0 and price < pivot_w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Filter"
timeframe = "6h"
leverage = 1.0
# ==========================================================
# End of strategy
# ==========================================================
```

## Last Updated
2026-04-27 12:23
