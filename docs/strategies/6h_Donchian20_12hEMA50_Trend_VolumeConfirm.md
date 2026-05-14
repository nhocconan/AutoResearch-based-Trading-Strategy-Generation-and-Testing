# Strategy: 6h_Donchian20_12hEMA50_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.420 | +8.6% | -8.2% | 267 | FAIL |
| ETHUSDT | 0.309 | +32.9% | -6.7% | 227 | PASS |
| SOLUSDT | 0.665 | +75.1% | -14.7% | 195 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.212 | +8.0% | -5.3% | 83 | PASS |
| SOLUSDT | 0.274 | +8.9% | -8.3% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses discrete sizing 0.25 to target 75-150 total trades over 4 years on 6h timeframe.
# Donchian channels provide clear breakout levels; 12h EMA50 ensures higher timeframe trend alignment.
# Volume confirmation filters breakouts with low participation. Designed for fewer, higher-quality trades
# to minimize fee drag while working in both bull and bear markets.

name = "6h_Donchian20_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) from prior candle only
    lookback_dc = 20
    prior_high_max = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    prior_low_min = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_dc, lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(prior_high_max[i]) or np.isnan(prior_low_min[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, close > 12h EMA50, volume spike
            if (high[i] > prior_high_max[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band, close < 12h EMA50, volume spike
            elif (low[i] < prior_low_min[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band OR volume drops below average
            if (low[i] < prior_low_min[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band OR volume drops below average
            if (high[i] > prior_high_max[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 22:17
