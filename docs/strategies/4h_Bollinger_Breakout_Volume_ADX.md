# Strategy: 4h_Bollinger_Breakout_Volume_ADX

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.540 | +50.6% | -14.4% | 212 | PASS |
| ETHUSDT | 0.136 | +26.7% | -13.7% | 207 | PASS |
| SOLUSDT | 0.346 | +48.6% | -18.5% | 199 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.170 | -5.1% | -8.1% | 68 | FAIL |
| ETHUSDT | 0.120 | +7.2% | -10.6% | 76 | PASS |
| SOLUSDT | -0.550 | -4.5% | -15.4% | 66 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Bollinger Band Breakout with Volume Confirmation and ADX Filter
Hypothesis: In trending markets (ADX > 20), price breaking above/below Bollinger Bands (20,2) 
with volume confirmation (volume > 1.5x average) indicates strong momentum. 
This strategy captures breakouts in both bull and bear markets while avoiding false signals 
in low-volume or low-volatility periods. Target: 20-40 trades/year to minimize fee drag.
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
    
    # Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 20,14,14)
    
    for i in range(start_idx, n):
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        adx_val = adx[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Strong trend (ADX > 20) and volume confirmation
            # Price breaks above upper BB = long
            if adx_val > 20 and price > upper and vol_conf:
                signals[i] = 0.25
                position = 1
            # Price breaks below lower BB = short
            elif adx_val > 20 and price < lower and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if trend weakens or price returns to middle band
            if adx_val < 15 or price < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if trend weakens or price returns to middle band
            if adx_val < 15 or price > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 05:12
