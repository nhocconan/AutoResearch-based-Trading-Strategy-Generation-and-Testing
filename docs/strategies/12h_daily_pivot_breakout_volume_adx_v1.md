# Strategy: 12h_daily_pivot_breakout_volume_adx_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.618 | +66.7% | -21.2% | 55 | PASS |
| ETHUSDT | -0.453 | -15.6% | -47.2% | 47 | FAIL |
| SOLUSDT | 0.760 | +137.5% | -49.7% | 37 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.605 | +14.1% | -8.9% | 24 | PASS |
| SOLUSDT | -0.066 | +2.1% | -16.4% | 19 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Pivot Breakout with Volume and ADX Filter v1
# Hypothesis: In ranging markets (common in 2025), price tends to breakout from daily pivot levels (S3/R3) with volume confirmation.
# We buy when price closes above S3 with volume and sell when price closes below R3 with volume.
# ADX filter ensures we only trade in trending conditions (ADX > 25) to avoid whipsaws in pure range.
# Works in both bull/bear as breakouts capture momentum in any trend.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "12h_daily_pivot_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    daily_range = daily_high - daily_low
    s1 = 2 * pivot - daily_high
    s2 = pivot - daily_range
    s3 = s2 - daily_range
    s4 = s3 - daily_range
    r1 = 2 * pivot - daily_low
    r2 = pivot + daily_range
    r3 = r2 + daily_range
    r4 = r3 + daily_range
    
    # Align pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    # ADX filter for trend strength
    # Calculate ADX using standard formula
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_ma / tr_ma)
    minus_di = 100 * (minus_dm_ma / tr_ma)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_filter = adx_values > 25  # Only trade when ADX > 25 (trending)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (stop) or reaches R3 (take profit)
            if close[i] <= s3_12h[i] or close[i] >= r3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above R3 (stop) or reaches S3 (take profit)
            if close[i] >= r3_12h[i] or close[i] <= s3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume and ADX confirmation
            if vol_filter[i] and adx_filter[i]:
                # Long breakout: price closes above S3
                if close[i] > s3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below R3
                elif close[i] < r3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 09:17
