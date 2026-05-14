# Strategy: 4h_GoldenRatio_StopReversal_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.171 | +0.9% | -24.5% | 1853 | DISCARD |
| ETHUSDT | 0.021 | +12.1% | -33.6% | 1897 | KEEP |
| SOLUSDT | 0.441 | +77.7% | -38.2% | 1884 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.547 | +19.2% | -13.4% | 610 | KEEP |
| SOLUSDT | 0.301 | +12.1% | -18.9% | 596 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
4h_GoldenRatio_StopReversal_12hTrend
Hypothesis: Uses golden ratio (61.8%) retracement levels from daily range for entries, aligned with 12h EMA50 trend, with stop-loss reversal mechanism. Works in both bull and bear markets by trading pullbacks to the golden ratio in trending conditions, with built-in risk management via stop reversals. Designed for low trade frequency (<30/year) to minimize fee burn.
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
    
    # Get daily data for golden ratio levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily high/low for golden ratio levels (61.8% retracement)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_range = daily_high - daily_low
    
    # Golden ratio levels: 61.8% retracement from daily low/high
    gr_upper = daily_low + (daily_range * 0.618)  # For uptrend: buy pullback to 61.8% from low
    gr_lower = daily_high - (daily_range * 0.618)  # For downtrend: sell pullback to 61.8% from high
    
    # Align to lower timeframe (4h) - values from previous day's close
    gr_upper_aligned = align_htf_to_ltf(prices, df_1d, gr_upper)
    gr_lower_aligned = align_htf_to_ltf(prices, df_1d, gr_lower)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    stop_price = 0.0  # Dynamic stop price
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(gr_upper_aligned[i]) or
            np.isnan(gr_lower_aligned[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Update stop price based on position
        if position == 1:  # Long position
            stop_price = daily_low_aligned[i]  # Stop at daily low
        elif position == -1:  # Short position
            stop_price = daily_high_aligned[i]  # Stop at daily high
        else:
            stop_price = 0.0
        
        # Check stop loss hit
        long_stop = position == 1 and low[i] <= stop_price
        short_stop = position == -1 and high[i] >= stop_price
        
        # Entry conditions: pullback to golden ratio level with trend alignment
        long_entry = (close[i] <= gr_upper_aligned[i]) and uptrend and (position <= 0)
        short_entry = (close[i] >= gr_lower_aligned[i]) and downtrend and (position >= 0)
        
        if long_stop or short_stop:
            # Stop hit: reverse position
            if long_stop:
                signals[i] = -0.30  # Reverse to short
                position = -1
            else:  # short_stop
                signals[i] = 0.30   # Reverse to long
                position = 1
        elif long_entry:
            signals[i] = 0.30
            position = 1
        elif short_entry:
            signals[i] = -0.30
            position = -1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_GoldenRatio_StopReversal_12hTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 03:59
