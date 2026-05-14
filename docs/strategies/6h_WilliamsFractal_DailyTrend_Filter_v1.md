# Strategy: 6h_WilliamsFractal_DailyTrend_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.147 | +26.9% | -15.6% | 84 | PASS |
| ETHUSDT | 0.049 | +21.3% | -16.5% | 102 | PASS |
| SOLUSDT | 1.139 | +192.2% | -26.3% | 92 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.819 | -0.0% | -5.6% | 29 | FAIL |
| ETHUSDT | 0.242 | +8.8% | -8.9% | 25 | PASS |
| SOLUSDT | -0.155 | +3.2% | -8.6% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_WilliamsFractal_DailyTrend_Filter_v1
Hypothesis: Daily Williams Fractal breakouts with 1d EMA50 trend filter on 6h timeframe.
Long when price breaks above bearish fractal (resistance) in uptrend (close > EMA50).
Short when price breaks below bullish fractal (support) in downtrend (close < EMA50).
Uses discrete sizing 0.25 to minimize fee churn. Williams fractals require 2-bar
confirmation delay on daily timeframe. Designed to work in both bull and bear markets
by following the daily trend while using fractals for precise entry/exit levels.
Target trades: 12-25/year (50-100 total over 4 years) to stay well below fee drag threshold.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams fractals and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams fractals on 1d (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar additional delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA (50) and fractal calculation (need 5 bars for fractals)
    start_idx = max(50, 5)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) in uptrend
            long_signal = (high_val > bearish_val) and (close_val > ema_50_1d_val)
            # Short: price breaks below bullish fractal (support) in downtrend
            short_signal = (low_val < bullish_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below bullish fractal (support) or trend reversal
            if close_val < bullish_val or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above bearish fractal (resistance) or trend reversal
            if close_val > bearish_val or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_DailyTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:00
