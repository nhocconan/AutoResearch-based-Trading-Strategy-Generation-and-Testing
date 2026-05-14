# Strategy: 4h_1d_ema_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.257 | +32.4% | -8.8% | 205 | PASS |
| ETHUSDT | 0.134 | +26.6% | -11.8% | 198 | PASS |
| SOLUSDT | 0.821 | +114.3% | -19.9% | 171 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.120 | -4.6% | -7.6% | 76 | FAIL |
| ETHUSDT | 0.477 | +13.3% | -9.1% | 63 | PASS |
| SOLUSDT | -0.018 | +5.0% | -10.8% | 58 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_ema_trend_volume_v1
# Hypothesis: Use 1d EMA for long-term trend direction, 4h EMA for medium-term trend confirmation,
# and volume surge for entry momentum. Trades only when both timeframes align with volume confirmation.
# Designed for 4h timeframe to target 20-50 trades/year by requiring multi-timeframe alignment and volume filter.
# Works in bull markets (trend following) and bear markets (avoids counter-trend trades when 1d trend opposes).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for medium-term trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for medium-term trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA(50) for long-term trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 2.0x average of last 24 periods (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA or loses upward momentum
            if close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA or loses downward momentum
            if close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above both EMAs with volume confirmation
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below both EMAs with volume confirmation
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 09:47
