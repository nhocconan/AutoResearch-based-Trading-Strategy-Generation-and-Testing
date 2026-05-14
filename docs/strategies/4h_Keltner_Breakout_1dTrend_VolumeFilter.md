# Strategy: 4h_Keltner_Breakout_1dTrend_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.215 | +30.0% | -10.1% | 115 | PASS |
| ETHUSDT | 0.504 | +49.6% | -12.7% | 105 | PASS |
| SOLUSDT | 0.726 | +91.6% | -21.9% | 96 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.856 | -1.3% | -6.9% | 43 | FAIL |
| ETHUSDT | 0.679 | +16.0% | -6.9% | 39 | PASS |
| SOLUSDT | 0.146 | +7.6% | -11.9% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_VolumeFilter
Hypothesis: Keltner Channel breakouts (2x ATR) with 1d trend alignment and volume confirmation capture strong momentum moves while filtering low-probability breakouts. Works in bull/bear via trend filter and avoids chop via ATR-based channel width. Designed for low trade frequency to minimize fee drag.
"""

name = "4h_Keltner_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period EMA on 1d close for trend filter
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate ATR(20) for Keltner Channel on 4h data
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA(20) of close for Keltner Channel middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands (2x ATR)
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: break above upper Keltner with volume spike and above 1d EMA20 (uptrend)
            if (close[i] > upper_keltner[i] and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower Keltner with volume spike and below 1d EMA20 (downtrend)
            elif (close[i] < lower_keltner[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below middle line (EMA20) or trend turns down
            if (close[i] < ema20[i] or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above middle line (EMA20) or trend turns up
            if (close[i] > ema20[i] or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 08:13
