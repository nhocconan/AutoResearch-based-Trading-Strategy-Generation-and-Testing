# Strategy: 4h_DailyPivot_Breakout_EMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.369 | +33.6% | -5.1% | 178 | PASS |
| ETHUSDT | 0.130 | +25.8% | -9.5% | 175 | PASS |
| SOLUSDT | 0.972 | +107.1% | -13.8% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.369 | -3.0% | -4.6% | 70 | FAIL |
| ETHUSDT | 0.332 | +9.7% | -6.7% | 62 | PASS |
| SOLUSDT | -0.177 | +3.7% | -7.7% | 53 | FAIL |

## Code
```python
# 4h_DailyPivot_Breakout_EMA50_Volume
# Hypothesis: Price breaks above/below daily pivot levels (R1/S1) with volume confirmation and trend filter (EMA50) capture institutional breakouts. Works in bull markets (breakouts continue) and bear markets (breakdowns continue). Uses daily timeframe for structure, 4h for execution. Target: 20-40 trades/year.
#!/usr/bin/env python3
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
    
    # Get daily data for pivot points and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 4h timeframe
    daily_pivot_4h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_4h[i]) or 
            np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_4h[i]
        price_below_s1 = close[i] < daily_s1_4h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and above daily EMA50
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and below daily EMA50
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below daily EMA50
            if (close[i] < daily_pivot_4h[i]) or (close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above daily EMA50
            if (close[i] > daily_pivot_4h[i]) or (close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_EMA50_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 10:21
