# Strategy: 6h_12h_1d_ema_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.271 | +32.4% | -10.9% | 54 | PASS |
| ETHUSDT | 0.139 | +26.8% | -16.8% | 48 | PASS |
| SOLUSDT | 1.055 | +145.4% | -13.1% | 37 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.933 | -3.9% | -10.4% | 22 | FAIL |
| ETHUSDT | 0.624 | +16.3% | -8.2% | 18 | PASS |
| SOLUSDT | -0.504 | -4.3% | -21.9% | 19 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_12h_1d_ema_trend_volume_v1
# Hypothesis: Use 1d EMA200 for trend direction, 12h EMA50 for intermediate trend, and 6h Donchian breakout for entry. Volume confirmation ensures institutional participation. Works in bull markets (trend-following breakouts) and bear markets (avoids counter-trend breakouts when higher timeframe trend opposes). Target: 15-30 trades/year per symbol (60-120 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_ema_trend_volume_v1"
timeframe = "6h"
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
    
    # Get 6h data for Donchian channels (entry signals)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 12h data for intermediate trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Calculate 12h EMA(50) for intermediate trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1d EMA(200) for long-term trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.8x average of last 48 periods (2 days in 6h)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h Donchian low or loses trend alignment
            if close[i] < donchian_low_aligned[i] or close[i] < ema_12h_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h Donchian high or loses trend alignment
            if close[i] > donchian_high_aligned[i] or close[i] > ema_12h_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 6h Donchian high with uptrend alignment and volume
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_12h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 6h Donchian low with downtrend alignment and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 09:56
