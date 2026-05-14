# Strategy: 4h_1D_Camarilla_Breakout_Volume_Confirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.403 | +47.5% | -11.9% | 48 | PASS |
| ETHUSDT | 0.031 | +16.5% | -18.9% | 43 | PASS |
| SOLUSDT | 0.821 | +167.4% | -32.3% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.235 | +2.1% | -9.1% | 18 | FAIL |
| ETHUSDT | 0.798 | +24.0% | -9.4% | 17 | PASS |
| SOLUSDT | -0.377 | -5.6% | -21.0% | 13 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_1D_Camarilla_Breakout_Volume_Confirmation
Hypothesis: Buy when price breaks above daily Camarilla H4 level with volume > 2.0x 50-period average, sell when price breaks below daily L4 level with volume confirmation. Uses 4h primary timeframe with 1d trend filter (price > EMA50 for longs, < EMA50 for shorts). Designed to work in both bull and bear markets by capturing genuine breakouts with strong volume, avoiding false breakouts in ranging markets. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 2.0)
    
    # Previous day's high/low/close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    camarilla_h4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily levels to 4h timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Daily EMA50 trend filter
    ema50_1d_raw = pd.Series(prev_close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above daily Camarilla H4 with volume expansion and price above daily EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_1d_aligned[i])
        
        # Short signal: break below daily Camarilla L4 with volume expansion and price below daily EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_1d_aligned[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1D_Camarilla_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 19:39
