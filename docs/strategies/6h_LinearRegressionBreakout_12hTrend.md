# Strategy: 6h_LinearRegressionBreakout_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.343 | +37.1% | -9.9% | 127 | PASS |
| ETHUSDT | 0.677 | +65.0% | -9.1% | 124 | PASS |
| SOLUSDT | 0.652 | +87.9% | -18.0% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.966 | -3.8% | -12.5% | 46 | FAIL |
| ETHUSDT | 0.464 | +12.9% | -6.7% | 39 | PASS |
| SOLUSDT | -0.162 | +2.7% | -13.0% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_LinearRegressionBreakout_12hTrend
Hypothesis: Price crossing above/below the 60-period linear regression channel with 12h trend filter and volume confirmation captures momentum moves while reducing false signals. Linear regression adapts to trend direction, and the channel acts as dynamic support/resistance. 12h trend filter ensures alignment with higher timeframe momentum. Volume confirmation adds conviction. Low frequency via 6h timeframe and strict entry criteria.
Target: 50-150 total trades over 4 years.
"""
name = "6h_LinearRegressionBreakout_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Linear Regression Channel (60-period)
    period = 60
    # Calculate linear regression slope and intercept using least squares
    # For each point, we need sum of x, y, x^2, xy over the window
    # Since x is just 0,1,2,...,period-1, we can precompute sums
    sum_x = period * (period - 1) / 2
    sum_x2 = (period - 1) * period * (2 * period - 1) / 6
    
    # Initialize arrays
    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    
    # Calculate using rolling window
    for i in range(period - 1, n):
        y_window = close[i - period + 1:i + 1]
        sum_y = np.sum(y_window)
        sum_xy = np.sum(y_window * np.arange(period))
        
        # Calculate slope and intercept
        slope[i] = (period * sum_xy - sum_x * sum_y) / (period * sum_x2 - sum_x * sum_x)
        intercept[i] = (sum_y - slope[i] * sum_x) / period
    
    # Calculate LR value at current point (end of window)
    lr_value = intercept + slope * (period - 1)
    
    # Calculate standard deviation of residuals for channel width
    residuals = close - lr_value
    std_dev = pd.Series(residuals).rolling(window=period, min_periods=period).std().values
    
    # Upper and lower channel (1 standard deviation)
    upper_channel = lr_value + std_dev
    lower_channel = lr_value - std_dev
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, period - 1)  # Need enough data for LR
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(lr_value[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper channel + 12h uptrend + volume
            if close[i] > upper_channel[i] and close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower channel + 12h downtrend + volume
            elif close[i] < lower_channel[i] and close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to linear regression value (mean reversion to trend)
            if position == 1:
                if close[i] <= lr_value[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= lr_value[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 06:43
