# Strategy: 4d_volume_price_action_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.152 | +27.9% | -20.8% | 90 | PASS |
| ETHUSDT | 0.288 | +41.9% | -18.4% | 84 | PASS |
| SOLUSDT | 1.315 | +439.9% | -30.4% | 76 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.943 | -7.8% | -14.0% | 35 | FAIL |
| ETHUSDT | 0.773 | +25.0% | -8.8% | 27 | PASS |
| SOLUSDT | 0.275 | +11.0% | -18.3% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4d_volume_price_action_v4
# Hypothesis: Uses 4-hour Donchian breakout with volume confirmation and 12-hour EMA trend filter.
# Enters long on Donchian breakout above in uptrend with volume spike; short on breakdown in downtrend with volume spike.
# Exits on opposite Donchian break or trend reversal. Designed for low trade frequency (~20-50/year) to minimize fee drag.
# Uses 12h EMA for stronger trend filter to reduce whipsaw and improve performance in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4d_volume_price_action_v4"
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
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h trend filter: EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_12h_aligned[i]
        daily_downtrend = close[i] < ema50_12h_aligned[i]
        
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
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: Donchian breakout or trend change
            if close[i] > donchian_high[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: Donchian breakout in uptrend
                if daily_uptrend and breakout_high:
                    position = 1
                    signals[i] = 0.30
                # Short entry: Donchian breakdown in downtrend
                elif daily_downtrend and breakout_low:
                    position = -1
                    signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-08 16:44
