# Strategy: 4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.464 | +37.7% | -5.3% | 185 | PASS |
| ETHUSDT | 0.328 | +35.5% | -8.9% | 183 | PASS |
| SOLUSDT | 0.870 | +96.6% | -13.8% | 148 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.814 | -6.5% | -6.6% | 80 | FAIL |
| ETHUSDT | 0.815 | +15.9% | -6.1% | 60 | PASS |
| SOLUSDT | -0.037 | +5.3% | -7.3% | 52 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout + Volume Spike + 1d EMA34 Trend Filter
Strategy: Enter long when price breaks above Camarilla H3 level with volume
          and price > 1d EMA34 (bullish trend). Enter short when price breaks
          below L3 level with volume and price < 1d EMA34 (bearish trend).
          Exit when price returns to Pivot level or trend weakens.
          Uses daily EMA34 as trend filter to avoid counter-trend trades.
          Designed for low trade frequency with clear breakout edge in both
          bull and bear markets. Camarilla levels derived from daily OHLC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily high, low, close for Camarilla levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels: Pivot = (H+L+C)/3
    # Range = H - L
    # H3 = Pivot + 1.1 * Range / 2
    # L3 = Pivot - 1.1 * Range / 2
    pivot = (daily_high + daily_low + daily_close) / 3.0
    rng = daily_high - daily_low
    h3 = pivot + 1.1 * rng / 2.0
    l3 = pivot - 1.1 * rng / 2.0
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        pivot_level = pivot_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: break above H3 with volume spike and above daily EMA34
            if (price > h3_level and volume_spike[i] and price > ema_34):
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume spike and below daily EMA34
            elif (price < l3_level and volume_spike[i] and price < ema_34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to pivot level or below EMA34 (trend change)
            if price <= pivot_level or price < ema_34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to pivot level or above EMA34 (trend change)
            if price >= pivot_level or price > ema_34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:09
