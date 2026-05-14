# Strategy: 6H_BollingerBreakout_1dEMA34Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.168 | +24.9% | -2.5% | 114 | PASS |
| ETHUSDT | 0.004 | +21.8% | -3.1% | 96 | PASS |
| SOLUSDT | -0.557 | +3.3% | -12.7% | 85 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.483 | -3.4% | -3.8% | 45 | FAIL |
| ETHUSDT | 1.014 | +13.8% | -2.0% | 46 | PASS |

## Code
```python
#%%
#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band breakout with 1d trend filter and volume confirmation.
In bull markets, price breaks above upper BB with upward 1d trend.
In bear markets, price breaks below lower BB with downward 1d trend.
Volume surge confirms institutional participation.
Designed for low trade frequency (12-30/year) to minimize fee drag.
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
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate 1d EMA34 trend
    ema34_daily = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB warmup
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB with bullish 1d trend and volume
            if (close[i] > bb_upper[i] and 
                close[i] > ema34_aligned[i] and  # Price above 1d EMA = bullish trend
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB with bearish 1d trend and volume
            elif (close[i] < bb_lower[i] and 
                  close[i] < ema34_aligned[i] and  # Price below 1d EMA = bearish trend
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to BB middle
            if position == 1:
                if close[i] < bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_BollingerBreakout_1dEMA34Trend_Volume"
timeframe = "6h"
leverage = 1.0
#%%
```

## Last Updated
2026-04-22 15:56
