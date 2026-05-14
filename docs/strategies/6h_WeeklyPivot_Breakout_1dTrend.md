# Strategy: 6h_WeeklyPivot_Breakout_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.193 | +28.4% | -9.8% | 103 | PASS |
| ETHUSDT | 0.156 | +27.6% | -7.1% | 98 | PASS |
| SOLUSDT | 0.899 | +122.2% | -17.4% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.900 | -1.3% | -6.8% | 36 | FAIL |
| ETHUSDT | 1.094 | +23.6% | -6.7% | 31 | PASS |
| SOLUSDT | 0.663 | +16.1% | -6.3% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_1dTrend
# Hypothesis: Use weekly pivot points from Monday's session to identify key support/resistance.
# Long when price breaks above weekly R1 with volume spike and 1d EMA50 uptrend.
# Short when price breaks below weekly S1 with volume spike and 1d EMA50 downtrend.
# Exit on mean reversion to weekly pivot (PP). Weekly pivots reset every Monday, providing
# fresh levels that adapt to market regime. Designed for low turnover (~15-25/year) to avoid fee drag.
# For 6h timeframe, we adapt to capture longer-term weekly structure.

name = "6h_WeeklyPivot_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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

    # Calculate weekly pivot points using Monday's OHLC
    # We'll compute weekly high/low/close from Friday's close to next Friday's close
    # But simpler: use 5-day (1 week) rolling window ending on Friday
    # Since we don't have day-of-week easily, approximate with 5-period (1 week) rolling on daily data
    # Instead: calculate pivots from prior week's daily OHLC
    # Get 1d data first
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from prior 5 trading days (approx 1 week)
    # Use rolling window of 5 on daily data
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot points
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above weekly R1 with volume spike and 1d EMA50 uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S1 with volume spike and 1d EMA50 downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below weekly pivot (mean reversion to center)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above weekly pivot
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 05:07
