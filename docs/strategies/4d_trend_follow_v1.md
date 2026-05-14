# Strategy: 4d_trend_follow_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.110 | +25.0% | -15.8% | 61 | PASS |
| ETHUSDT | -0.550 | -18.3% | -33.6% | 68 | FAIL |
| SOLUSDT | 0.699 | +123.5% | -30.2% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.207 | +2.8% | -6.8% | 21 | FAIL |
| SOLUSDT | 0.263 | +10.5% | -15.4% | 19 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4d_trend_follow_v1
# Hypothesis: Uses 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation.
# Enters long on Donchian breakout above in uptrend with volume spike; short on breakdown in downtrend with volume spike.
# Exits on opposite Donchian break or trend reversal. Designed for low trade frequency (~20-50/year) to minimize fee drag.
# Works in both bull and bear markets via trend filter that aligns with higher timeframe direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4d_trend_follow_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_high = close[i] > donchian_high[i-1]
        breakout_low = close[i] < donchian_low[i-1]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Donchian breakdown or trend change
            if close[i] < donchian_low[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian breakout or trend change
            if close[i] > donchian_high[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: Donchian breakout in uptrend
                if daily_uptrend and breakout_high:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Donchian breakdown in downtrend
                elif daily_downtrend and breakout_low:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 16:28
