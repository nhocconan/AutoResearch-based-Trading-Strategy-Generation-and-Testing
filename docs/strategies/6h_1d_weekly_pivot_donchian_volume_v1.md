# Strategy: 6h_1d_weekly_pivot_donchian_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.056 | +17.0% | -11.6% | 156 | FAIL |
| ETHUSDT | 0.422 | +46.3% | -9.8% | 144 | PASS |
| SOLUSDT | 0.656 | +92.6% | -22.4% | 140 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.391 | +11.9% | -8.8% | 48 | PASS |
| SOLUSDT | 0.355 | +11.2% | -8.6% | 49 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Weekly pivots from daily data provide institutional reference points for breakout validation
# Donchian breakouts capture momentum; 1d weekly pivot direction filters for structure-aligned moves
# Volume confirmation ensures breakout authenticity
# Works in bull/bear: weekly pivot direction adapts to higher timeframe bias
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_weekly_pivot_donchian_volume_v1"
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
    
    # Load 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily OHLC (using prior week's data)
    # Weekly pivot: (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # Weekly R1: (2 * Pivot) - Prior Week Low
    # Weekly S1: (2 * Pivot) - Prior Week High
    # We use the pivot from the completed week prior to current 6h bar
    
    # Resample daily to weekly using actual weekly boundaries (not resample)
    # Since we can't resample, we calculate weekly pivot from rolling weekly window
    # Using min_periods to ensure we only use completed weeks
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values  # 5 trading days
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly pivot data to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < weekly pivot (trend change vs weekly structure)
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > weekly pivot (trend change vs weekly structure)
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + weekly pivot filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > weekly pivot (bullish breakout above weekly structure)
                if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < weekly pivot (bearish breakout below weekly structure)
                elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 12:10
