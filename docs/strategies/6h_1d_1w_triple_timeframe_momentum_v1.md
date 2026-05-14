# Strategy: 6h_1d_1w_triple_timeframe_momentum_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.266 | -11.8% | -18.8% | 458 | FAIL |
| ETHUSDT | -0.317 | +9.5% | -8.4% | 400 | FAIL |
| SOLUSDT | 0.489 | +53.5% | -12.9% | 357 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.260 | +7.9% | -6.5% | 58 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_1d_1w_triple_timeframe_momentum_v1
# Hypothesis: Combining 60-period EMA trend filter from 60-day (1d) timeframe with 
# 6-hour momentum (close > open) and volume confirmation creates a robust trend-following
# strategy that works in both bull and bear markets. The 60-day EMA captures the 
# primary trend direction, while 6-hour momentum and volume filters ensure entries 
# occur only during strong momentum bursts with institutional participation, 
# reducing false signals and whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_triple_timeframe_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (60-day EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 60-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema60_1d = pd.Series(close_1d).ewm(span=60, min_periods=60, adjust=False).mean().values
    ema60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema60_1d)
    
    # Get weekly data for additional trend confirmation (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        # If weekly data not available, continue with daily only
        ema30_1w_aligned = np.full(n, np.nan)
    else:
        close_1w = df_1w['close'].values
        ema30_1w = pd.Series(close_1w).ewm(span=30, min_periods=30, adjust=False).mean().values
        ema30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema30_1w)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    # 6-hour momentum: close > open (bullish candle)
    bullish_momentum = close > prices['open'].values
    bearish_momentum = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema60_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Optional weekly trend filter (use if available)
        weekly_filter_long = True
        weekly_filter_short = True
        if not np.isnan(ema30_1w_aligned[i]):
            weekly_filter_long = close[i] > ema30_1w_aligned[i]
            weekly_filter_short = close[i] < ema30_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 60-day EMA or loses bullish momentum
            if close[i] < ema60_1d_aligned[i] or not bullish_momentum[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 60-day EMA or loses bearish momentum
            if close[i] > ema60_1d_aligned[i] or not bearish_momentum[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above 60-day EMA, bullish momentum, volume confirmation
            if (close[i] > ema60_1d_aligned[i] and 
                bullish_momentum[i] and 
                vol_confirm[i] and 
                weekly_filter_long):
                position = 1
                signals[i] = 0.25
            # Short entry: price below 60-day EMA, bearish momentum, volume confirmation
            elif (close[i] < ema60_1d_aligned[i] and 
                  bearish_momentum[i] and 
                  vol_confirm[i] and 
                  weekly_filter_short):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 09:34
