# Strategy: 4h_ThreeBarBreakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.248 | +29.9% | -7.4% | 116 | PASS |
| ETHUSDT | 0.055 | +22.5% | -8.3% | 124 | PASS |
| SOLUSDT | 0.600 | +69.9% | -21.8% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.556 | +2.2% | -5.3% | 45 | FAIL |
| ETHUSDT | 0.060 | +6.4% | -9.8% | 43 | PASS |
| SOLUSDT | 0.144 | +7.5% | -7.6% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_ThreeBarBreakout_Volume_Trend
# Hypothesis: Three consecutive bullish/bearish candles in the direction of the 1-day trend,
# with volume confirmation, capture momentum bursts while avoiding false breakouts.
# Designed for low trade frequency (~30-50/year) to minimize fee drag and work in both bull and bear markets.
# Uses 1-day EMA for trend filter and volume spike confirmation for institutional participation.
timeframe = "4h"
name = "4h_ThreeBarBreakout_Volume_Trend"
leverage = 1.0

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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA trend filter (21-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Three-bar bullish pattern: three consecutive closes higher than prior
            bullish = (close[i] > close[i-1] and 
                       close[i-1] > close[i-2] and 
                       close[i-2] > close[i-3])
            # Three-bar bearish pattern: three consecutive closes lower than prior
            bearish = (close[i] < close[i-1] and 
                       close[i-1] < close[i-2] and 
                       close[i-2] < close[i-3])
            
            # Long: three bullish bars + above 1d EMA + volume spike
            if bullish and close[i] > ema_1d_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: three bearish bars + below 1d EMA + volume spike
            elif bearish and close[i] < ema_1d_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: three consecutive bearish bars or price drops below 1d EMA
            bearish_exit = (close[i] < close[i-1] and 
                            close[i-1] < close[i-2] and 
                            close[i-2] < close[i-3])
            if bearish_exit or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: three consecutive bullish bars or price rises above 1d EMA
            bullish_exit = (close[i] > close[i-1] and 
                            close[i-1] > close[i-2] and 
                            close[i-2] > close[i-3])
            if bullish_exit or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 01:36
