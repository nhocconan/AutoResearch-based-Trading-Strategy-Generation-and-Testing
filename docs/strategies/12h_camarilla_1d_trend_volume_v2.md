# Strategy: 12h_camarilla_1d_trend_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.282 | +11.3% | -10.8% | 161 | FAIL |
| ETHUSDT | 0.179 | +28.0% | -13.8% | 139 | PASS |
| SOLUSDT | 0.070 | +21.3% | -24.3% | 131 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.419 | +10.8% | -5.5% | 50 | PASS |
| SOLUSDT | -0.594 | -2.0% | -11.0% | 48 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v2
# Hypothesis: 12h strategy using 1d Camarilla H4/L4 levels with 12h trend filter and volume confirmation.
# H4/L4 are stronger reversal levels than H3/L3, reducing false breakouts. Volume > 1.5x 20-period average
# filters weak moves. Trend filter uses 12h EMA(50) alignment. Discrete sizing (±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in both bull and bear via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA(50) for trend bias
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # Handle first bar
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Camarilla pivot point
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla H4 and L4 levels (stronger reversal levels)
    h4 = pivot_point + (range_1d * 1.1 / 2)
    l4 = pivot_point - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back below H4 (profit taken or reversal)
            if close[i] < h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back above L4 (profit taken or reversal)
            if close[i] > l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price crosses above H4 with 12h bullish bias (price above EMA50)
                if close[i] > h4_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below L4 with 12h bearish bias (price below EMA50)
                elif close[i] < l4_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 04:52
