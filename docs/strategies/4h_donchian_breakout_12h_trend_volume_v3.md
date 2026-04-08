# Strategy: 4h_donchian_breakout_12h_trend_volume_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.100 | +24.6% | -14.5% | 182 | PASS |
| ETHUSDT | 0.200 | +30.8% | -11.5% | 169 | PASS |
| SOLUSDT | 0.817 | +116.8% | -28.1% | 171 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.324 | +3.1% | -5.7% | 62 | FAIL |
| ETHUSDT | 0.523 | +13.3% | -9.4% | 57 | PASS |
| SOLUSDT | 1.092 | +24.4% | -9.5% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume + ATR Stop v3
Hypothesis: Donchian breakouts capture strong trends. Filter by 12h EMA trend (more stable than 1d) and volume confirmation. ATR-based stop manages risk. Works in bull/bear by using volatility-adjusted stops and trend alignment. Targets 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = df_12h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h ATR(20) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 4h Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend reverses OR ATR stop
            if (close[i] <= lowest_low[i] or 
                close[i] < ema_50_12h_aligned[i] or
                close[i] <= (highest_high[i-1] - 2.5 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend reverses OR ATR stop
            if (close[i] >= highest_high[i] or 
                close[i] > ema_50_12h_aligned[i] or
                close[i] >= (lowest_low[i-1] + 2.5 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] > highest_high[i-1] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] < lowest_low[i-1] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 00:51
