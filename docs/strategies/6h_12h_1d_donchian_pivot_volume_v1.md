# Strategy: 6h_12h_1d_donchian_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.146 | +26.7% | -11.0% | 111 | PASS |
| ETHUSDT | 0.381 | +41.9% | -10.9% | 98 | PASS |
| SOLUSDT | 0.665 | +88.7% | -20.3% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.050 | -3.1% | -10.0% | 41 | FAIL |
| ETHUSDT | 0.454 | +12.4% | -6.9% | 34 | PASS |
| SOLUSDT | 0.217 | +8.6% | -7.0% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian breakout + 1d weekly pivot direction + volume confirmation
# Donchian(20) breakout from 12h captures medium-term trend momentum
# Weekly pivot from 1d provides institutional bias (long above weekly PP, short below)
# Volume confirmation ensures breakout validity
# Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear markets: breakout follows trends, pivot filter adapts to regime

name = "6h_12h_1d_donchian_pivot_volume_v1"
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
    open_time = prices['open_time'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Load 1d data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly high = max of prior 5 daily highs
    # Weekly low = min of prior 5 daily lows
    # Weekly close = prior daily close (Friday's close)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).shift(1).rolling(window=5, min_periods=5).last().values  # Prior Friday's close
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align 1d weekly pivot to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    
    # Pre-compute session filter (08-20 UTC) - optional but helpful
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian low or below weekly PP
            if close[i] < donchian_low_aligned[i] or close[i] < weekly_pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian high or above weekly PP
            if close[i] > donchian_high_aligned[i] or close[i] > weekly_pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Donchian breakout with volume confirmation and pivot filter
            if close[i] > donchian_high_aligned[i] and volume_confirmed and close[i] > weekly_pp_aligned[i]:
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low_aligned[i] and volume_confirmed and close[i] < weekly_pp_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 16:41
