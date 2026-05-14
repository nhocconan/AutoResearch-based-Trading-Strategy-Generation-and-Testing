# Strategy: 4h_RangeBreakout_Volume_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.101 | +13.3% | -16.4% | 142 | FAIL |
| ETHUSDT | 0.440 | +51.9% | -13.9% | 129 | PASS |
| SOLUSDT | 0.936 | +163.5% | -23.8% | 118 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.003 | +5.0% | -12.6% | 50 | PASS |
| SOLUSDT | 0.062 | +5.9% | -14.1% | 42 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_RangeBreakout_Volume_Trend_Filter
Hypothesis: In ranging markets, price tends to break out of recent highs/lows with volume confirmation.
Uses Donchian breakout (20-period) on 4h timeframe with volume spike (>2x 20-bar average) and 
trend filter (4h EMA50) to avoid false breakouts. Works in both bull and bear markets by
capturing momentum bursts. Target: 25-40 trades/year.
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
    
    # Get 4h data for Donchian calculation (self-referential but using current timeframe)
    # We'll calculate Donchian directly on 4h prices
    
    # Calculate Donchian channels (20-period) - using current high/low
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 2.0)
    
    # Trend filter: EMA50 on 4h
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and uptrend
            if (close[i] > donchian_high[i] and volume_spike[i] and close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and downtrend
            elif (close[i] < donchian_low[i] and volume_spike[i] and close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to Donchian low or trend fails
            if (close[i] <= donchian_low[i] or close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian high or trend fails
            if (close[i] >= donchian_high[i] or close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RangeBreakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 17:30
