# Strategy: 4h_bollinger_breakout_volume_adx_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.292 | +34.8% | -13.9% | 183 | PASS |
| ETHUSDT | 0.199 | +30.8% | -12.1% | 178 | PASS |
| SOLUSDT | 0.355 | +48.7% | -24.6% | 167 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.169 | -4.8% | -7.5% | 61 | FAIL |
| ETHUSDT | 0.362 | +11.1% | -9.8% | 62 | PASS |
| SOLUSDT | -0.661 | -5.6% | -15.1% | 55 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Bollinger Breakout with Volume and ADX Trend Filter
Long when price breaks above upper Bollinger Band with volume surge and ADX > 25
Short when price breaks below lower Bollinger Band with volume surge and ADX > 25
Exit when price returns to middle Bollinger Band
Designed to capture volatility breakouts in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Bollinger Bands (20, 2) ===
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # === ADX (14) for trend strength ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed averages
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / (tr_ma + 1e-10)
    minus_di = 100 * minus_dm_ma / (tr_ma + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(basis[i]) or np.isnan(adx[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle Bollinger Band
            if close[i] <= basis[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle Bollinger Band
            if close[i] >= basis[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume surge and strong trend
            if vol_ratio[i] < 1.5 or adx[i] < 25:
                signals[i] = 0.0
                continue
            
            # Entry: Bollinger Band breakout with volume and trend confirmation
            if close[i] > upper[i]:
                # Price above upper band -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower[i]:
                # Price below lower band -> short
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 23:14
