# Strategy: 12h_daily_ema_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.019 | +16.3% | -14.8% | 61 | FAIL |
| ETHUSDT | -0.626 | -24.5% | -32.0% | 76 | FAIL |
| SOLUSDT | 0.685 | +119.6% | -34.0% | 53 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.052 | +4.9% | -18.1% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
# 12h_daily_ema_trend_volume_v1
# Hypothesis: 12h strategy using daily EMA trend with volume confirmation. Long when price > daily EMA50 with volume > 1.5x 20-period average. Short when price < daily EMA50 with volume > 1.5x 20-period average. Exit on opposite cross. Uses daily trend for direction, 12h for execution, volume for confirmation. Target: 15-30 trades/year (60-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_ema_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below daily EMA50
            if close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above daily EMA50
            if close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for EMA cross with volume confirmation
            bullish_cross = (close[i] > ema_50_aligned[i]) and volume_confirmed
            bearish_cross = (close[i] < ema_50_aligned[i]) and volume_confirmed
            
            if bullish_cross:
                position = 1
                signals[i] = 0.25
            elif bearish_cross:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 00:54
