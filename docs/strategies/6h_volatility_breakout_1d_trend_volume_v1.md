# Strategy: 6h_volatility_breakout_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.116 | +16.4% | -9.1% | 90 | FAIL |
| ETHUSDT | -0.053 | +17.7% | -11.8% | 89 | FAIL |
| SOLUSDT | 0.339 | +39.6% | -12.2% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.303 | +9.7% | -6.9% | 24 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_volatility_breakout_1d_trend_volume_v1
# Hypothesis: Combine Bollinger Band volatility breakout with daily trend filter and volume confirmation.
# In low volatility (BB squeeze), wait for breakout in direction of daily EMA trend.
# Volume confirms breakout strength. Works in both bull/bear markets by following higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_volatility_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = ma + (std * bb_std)
    lower = ma - (std * bb_std)
    bb_width = upper - lower
    
    # Bollinger Band squeeze: width < 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    squeeze = bb_width < bb_width_ma
    
    # Daily EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 24-period average (4 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(bb_period, 20, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(squeeze[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle band or trend fails
            if close[i] < ma[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band or trend fails
            if close[i] > ma[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade after Bollinger Band squeeze (low volatility)
            if squeeze[i-1] and volume_filter:  # squeeze was present on previous bar
                # Breakout long: price breaks above upper band with uptrend
                if close[i] > upper[i] and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below lower band with downtrend
                elif close[i] < lower[i] and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 08:13
