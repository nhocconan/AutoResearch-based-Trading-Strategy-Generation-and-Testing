# Strategy: 6h_Donchian20_1wEMA50_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.150 | +16.7% | -11.8% | 65 | FAIL |
| ETHUSDT | 0.082 | +23.8% | -11.8% | 51 | PASS |
| SOLUSDT | 0.180 | +29.4% | -16.6% | 36 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.124 | +7.1% | -7.5% | 18 | PASS |
| SOLUSDT | -0.482 | +1.3% | -8.6% | 14 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Uses Donchian channels from prior 6h for breakout structure, 1w EMA50 for trend filter
# Volume confirmation (>1.8x 20 EMA) ensures breakout has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Weekly EMA50 ensures we only trade with the major trend, reducing whipsaw in ranging markets.

name = "6h_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels (20-period) from prior completed 6h bar
    # We need to calculate this on 6h data, but we can use the current prices with a 20-bar lookback
    # Since we're in the generate_signals loop, we'll precompute the 6h Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + price above 1w EMA50 + volume spike
            if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + price below 1w EMA50 + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1w EMA50
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1w EMA50
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 07:51
