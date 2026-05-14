# Strategy: 4h_1D_Camarilla_Breakout_Volume_Spike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.141 | +27.1% | -13.3% | 26 | PASS |
| ETHUSDT | -0.581 | -26.3% | -39.3% | 25 | FAIL |
| SOLUSDT | 0.446 | +73.1% | -39.1% | 24 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.113 | +7.1% | -7.7% | 10 | PASS |
| SOLUSDT | 0.072 | +5.5% | -21.4% | 7 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1D_Camarilla_Breakout_Volume_Spike_v1
Hypothesis: Enter long when price breaks above daily Camarilla H4 with volume > 3x 30-day average and short when breaks below daily L4 with volume > 3x 30-day average. Use daily EMA50 as trend filter (only long when price > EMA50, short when price < EMA50). Target low trade frequency (<40/year) with high win rate by requiring strong volume spike and trend alignment. Designed to work in both bull and bear markets by capturing genuine breakouts with institutional volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 3x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_spike = volume > (vol_ma_30 * 3.0)
    
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
    
    for i in range(35, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above daily Camarilla H4 with volume spike and price above daily EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_spike[i] and 
                      close[i] > ema50_1d_aligned[i])
        
        # Short signal: break below daily Camarilla L4 with volume spike and price below daily EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_spike[i] and 
                       close[i] < ema50_1d_aligned[i])
        
        if position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price breaks below L4 with volume confirmation
            if short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price breaks above H4 with volume confirmation
            if long_signal:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1D_Camarilla_Breakout_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 19:48
