# Strategy: 1h_DailyPivot_Breakout_EMA20_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.443 | +4.6% | -9.9% | 525 | FAIL |
| ETHUSDT | 0.118 | +25.5% | -9.7% | 506 | PASS |
| SOLUSDT | 0.615 | +74.8% | -21.8% | 497 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.570 | +12.7% | -7.7% | 154 | PASS |
| SOLUSDT | 0.269 | +9.4% | -7.4% | 164 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 1h timeframe
    daily_pivot_1h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_1h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_1h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily EMA20 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need daily EMA20, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_1h[i]) or 
            np.isnan(daily_r1_1h[i]) or 
            np.isnan(daily_s1_1h[i]) or 
            np.isnan(ema20_1h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA20
        price_above_ema = close[i] > ema20_1h[i]
        price_below_ema = close[i] < ema20_1h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_1h[i]
        price_below_s1 = close[i] < daily_s1_1h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and above daily EMA20
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below daily S1 with volume and below daily EMA20
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below daily EMA20
            if (close[i] < daily_pivot_1h[i]) or (close[i] < ema20_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above daily EMA20
            if (close[i] > daily_pivot_1h[i]) or (close[i] > ema20_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DailyPivot_Breakout_EMA20_Volume"
timeframe = "1h"
leverage = 1.0
```

## Last Updated
2026-04-17 10:21
