# Strategy: 4h_Camarilla_H4L4_1dEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.191 | +29.2% | -9.3% | 140 | PASS |
| ETHUSDT | 0.149 | +27.6% | -13.1% | 139 | PASS |
| SOLUSDT | 0.863 | +125.8% | -17.2% | 113 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.873 | -3.2% | -8.2% | 52 | FAIL |
| ETHUSDT | 1.582 | +36.2% | -6.5% | 41 | PASS |
| SOLUSDT | 0.001 | +5.3% | -9.5% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation.
- Camarilla H4/L4 levels from 1d provide strong intraday pivot points for 4h breakouts.
- 1d EMA34 trend filter ensures alignment with daily momentum (works in bull/bear via trend alignment).
- Volume spike (>2.0x 20-period average) confirms breakout validity with higher threshold to reduce whipsaws and trade frequency.
- Discrete position sizing (0.25) balances return and drawdown control.
- Target: 80-160 total trades over 4 years (20-40/year) on 4h timeframe.
- Uses 1d HTF data loaded ONCE before loop per MTF rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla H4, L4 levels: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (using previous completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average volume (4h * 5 = ~20h, close to 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H4 with volume spike and above 1d EMA34
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L4 with volume spike and below 1d EMA34
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L4 OR below 1d EMA34
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H4 OR above 1d EMA34
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 02:27
