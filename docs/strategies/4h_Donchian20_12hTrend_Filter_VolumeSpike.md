# Strategy: 4h_Donchian20_12hTrend_Filter_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.218 | +32.9% | -14.0% | 192 | PASS |
| ETHUSDT | 0.244 | +36.5% | -19.9% | 186 | PASS |
| SOLUSDT | 1.329 | +395.8% | -29.8% | 176 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.254 | -11.3% | -12.9% | 71 | FAIL |
| ETHUSDT | 0.295 | +11.1% | -12.6% | 59 | PASS |
| SOLUSDT | -0.099 | +1.3% | -16.9% | 63 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Donchian20_12hTrend_Filter_VolumeSpike
Hypothesis: Uses 20-period Donchian channel breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
This combines trend following with volatility breakouts, designed to work in both trending and ranging markets by
only taking breakouts in the direction of the 12h trend. Volume spikes filter false breakouts. Targets ~25-35 trades/year.
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
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > high_max_20[i-1]  # Break above previous high
        breakout_down = low[i] < low_min_20[i-1]  # Break below previous low
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic: Only take breakouts in direction of 12h trend
        long_entry = breakout_up and trend_up and vol_confirm
        short_entry = breakout_down and trend_down and vol_confirm
        
        # Exit logic: Opposite Donchian breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
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
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hTrend_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 02:23
