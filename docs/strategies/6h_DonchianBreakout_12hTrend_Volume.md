# Strategy: 6h_DonchianBreakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.302 | +3.5% | -16.8% | 72 | FAIL |
| ETHUSDT | 0.006 | +17.8% | -17.7% | 75 | PASS |
| SOLUSDT | 1.156 | +228.9% | -23.4% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.321 | +11.2% | -10.0% | 24 | PASS |
| SOLUSDT | -0.179 | +0.6% | -15.9% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_DonchianBreakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_12h = close_12h > ema50_12h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Donchian channel on 6h (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_up_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Donchian breakout + 12h uptrend + volume confirmation
            if close[i] > donchian_high[i-1] and trend_up_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + 12h downtrend + volume confirmation
            elif close[i] < donchian_low[i-1] and not trend_up_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR 12h trend turns down
            if close[i] < donchian_low[i-1] or not trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout OR 12h trend turns up
            if close[i] > donchian_high[i-1] or trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 18:30
