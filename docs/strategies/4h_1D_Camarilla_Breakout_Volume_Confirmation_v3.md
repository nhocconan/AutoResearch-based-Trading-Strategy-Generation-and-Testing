# Strategy: 4h_1D_Camarilla_Breakout_Volume_Confirmation_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.393 | +47.0% | -11.5% | 52 | PASS |
| ETHUSDT | 0.355 | +48.5% | -17.5% | 45 | PASS |
| SOLUSDT | 0.804 | +162.3% | -32.3% | 44 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.107 | +3.9% | -9.0% | 21 | FAIL |
| ETHUSDT | 0.889 | +26.5% | -8.9% | 17 | PASS |
| SOLUSDT | -0.003 | +3.5% | -16.5% | 13 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_1D_Camarilla_Breakout_Volume_Confirmation_v3
Hypothesis: Buy when price breaks above daily Camarilla H4 level with volume > 1.8x 50-period average and price > daily EMA50, sell when price breaks below daily L4 level with volume confirmation and price < daily EMA50. Uses 4h primary timeframe with 1d trend filter. Added minimum holding period of 8 bars (32 hours) to reduce overtrading and filter false breakouts. Designed to work in both bull and bear markets by capturing genuine breakouts with strong volume and trend alignment. Slightly relaxed volume threshold from 2.0x to 1.8x to increase trade frequency while maintaining quality.
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
    
    # Volume confirmation: current volume > 1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.8)
    
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
    bars_since_entry = 0  # Track holding period
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: break above daily Camarilla H4 with volume expansion and price above daily EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_1d_aligned[i])
        
        # Short signal: break below daily Camarilla L4 with volume expansion and price below daily EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_1d_aligned[i])
        
        # Exit conditions: minimum holding period reached and opposite signal
        if position == 1 and bars_since_entry >= 8 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 8 and long_signal:
            position = 1
            signals[i] = position_size
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1D_Camarilla_Breakout_Volume_Confirmation_v3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 19:41
