# Strategy: 4h_Camarilla_H4L4_Breakout_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.880 | +56.0% | -7.1% | 273 | PASS |
| ETHUSDT | 0.531 | +45.7% | -7.5% | 261 | PASS |
| SOLUSDT | 0.440 | +51.3% | -19.8% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.796 | -6.8% | -7.2% | 109 | FAIL |
| ETHUSDT | 1.318 | +23.2% | -8.0% | 92 | PASS |
| SOLUSDT | 0.809 | +16.0% | -6.2% | 71 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H4L4_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: On 4h timeframe, Camarilla H4/L4 breakout from prior 1d bar combined with 1d EMA34 trend filter and volume spike confirmation.
H4/L4 levels represent stronger support/resistance than H3/L3, reducing false breakouts.
1d EMA34 provides a reliable long-term trend filter that works in both bull and bear markets.
Volume spike confirms institutional participation, reducing whipsaw.
Designed for 19-50 trades/year (75-200 over 4 years) to minimize fee drag.
Works in bull markets via breakout continuation and bear markets via mean reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4, L4 (stronger intraday support/resistance than H3/L3)
    camarilla_range = prev_high - prev_low
    h4 = prev_close + camarilla_range * 1.1 / 2
    l4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H4/L4 breakout + volume spike + 1d EMA34 trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA34
            long_trend = curr_close > ema_34_1d_aligned[i]
            short_trend = curr_close < ema_34_1d_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla H4 (failed breakout) or trend reverses
            if curr_close < h4_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla L4 (failed breakout) or trend reverses
            if curr_close > l4_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 09:07
