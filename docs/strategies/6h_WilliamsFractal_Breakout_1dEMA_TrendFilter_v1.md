# Strategy: 6h_WilliamsFractal_Breakout_1dEMA_TrendFilter_v1

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
6h_WilliamsFractal_Breakout_1dEMA_TrendFilter_v1
Hypothesis: Trade breakouts of daily Williams fractals on 6h timeframe with 1d EMA50 trend filter.
In bull markets: buy when price breaks above the most recent bullish fractal (resistance) and price > EMA50.
In bear markets: sell when price breaks below the most recent bearish fractal (support) and price < EMA50.
Fractals require 2-bar confirmation (additional_delay_bars=2) to avoid false signals.
Exit on opposite fractal touch or trend reversal.
Position size: 0.25 to limit drawdown in volatile markets.
Target: 12-25 trades/year to stay well under 300-trade 6h hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter and Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 bars for fractals
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams fractals on 1d data (requires 2-bar confirmation delay)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Align with additional_delay_bars=2 for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above most recent bullish fractal + 1d uptrend
            long_setup = (close[i] > bullish_fractal_aligned[i]) and htf_1d_bullish
            
            # Short setup: price breaks below most recent bearish fractal + 1d downtrend
            short_setup = (close[i] < bearish_fractal_aligned[i]) and htf_1d_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches bearish fractal (stop) OR 1d trend turns bearish
            if (close[i] <= bearish_fractal_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches bullish fractal (stop) OR 1d trend turns bullish
            if (close[i] >= bullish_fractal_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 15:58
