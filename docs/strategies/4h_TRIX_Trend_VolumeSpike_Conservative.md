# Strategy: 4h_TRIX_Trend_VolumeSpike_Conservative

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.438 | +39.3% | -10.6% | 160 | PASS |
| ETHUSDT | 0.410 | +41.5% | -7.9% | 159 | PASS |
| SOLUSDT | 0.237 | +35.6% | -22.1% | 141 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.362 | +2.9% | -6.3% | 57 | FAIL |
| ETHUSDT | 0.037 | +5.9% | -7.8% | 65 | PASS |
| SOLUSDT | -0.118 | +3.5% | -10.2% | 55 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_TRIX_Trend_VolumeSpike_Conservative
Hypothesis: TRIX momentum crossover combined with 1d EMA100 trend filter and volume spikes captures high-probability trend continuations. Conservative settings (TRIX crossover only after 2-bar confirmation) reduce overtrading while maintaining edge in both bull and bear markets. Targets 15-25 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of ROC)
    # ROC period = 1
    roc = np.diff(close, prepend=close[0])
    # Three consecutive EMAs
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # Scale for readability
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: >1.8x 20-period MA (approx 10 hours on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(trix_signal[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA100
        uptrend = close[i] > ema_100_1d_aligned[i]
        downtrend = close[i] < ema_100_1d_aligned[i]
        
        # TRIX crossover with confirmation (require 2 consecutive bars)
        trix_bullish = trix[i] > trix_signal[i] and trix[i-1] > trix_signal[i-1]
        trix_bearish = trix[i] < trix_signal[i] and trix[i-1] < trix_signal[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Entry logic: TRIX crossover in direction of trend with volume
        long_entry = vol_confirm and uptrend and trix_bullish
        short_entry = vol_confirm and downtrend and trix_bearish
        
        # Exit logic: opposite TRIX crossover or trend change
        long_exit = trix_bearish or (not uptrend)
        short_exit = trix_bullish or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TRIX_Trend_VolumeSpike_Conservative"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 03:18
