# Strategy: 4h_KeltnerBreakout_12hTrend_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.193 | +30.5% | -16.3% | 89 | PASS |
| ETHUSDT | 0.299 | +40.6% | -14.4% | 84 | PASS |
| SOLUSDT | 1.110 | +233.6% | -28.2% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.326 | -8.5% | -12.9% | 35 | FAIL |
| ETHUSDT | 0.264 | +10.1% | -9.2% | 31 | PASS |
| SOLUSDT | -0.166 | +1.0% | -19.4% | 28 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_KeltnerBreakout_12hTrend_VolumeFilter
Hypothesis: Keltner Channel breakouts with 12h EMA trend filter and volume spikes capture momentum moves while avoiding false breakouts. The Keltner Channel adapts to volatility, making it effective in both bull and bear markets. Volume confirmation ensures momentum, and the 12h EMA filter ensures we trade with the higher timeframe trend. Targets 20-40 trades/year.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Keltner Channel (20-period EMA, 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # Volume confirmation: >1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions: break of Keltner Channel
        breakout_upper = close[i] > kc_upper[i]
        breakdown_lower = close[i] < kc_lower[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_upper
        short_entry = vol_confirm and downtrend and breakdown_lower
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_lower or (not uptrend)
        short_exit = breakout_upper or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KeltnerBreakout_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 03:13
