# Strategy: 4h_Camarilla_H3L3_Breakout_VolumeSpike_ATRFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.393 | -0.7% | -13.9% | 168 | FAIL |
| ETHUSDT | 0.240 | +34.2% | -13.0% | 146 | PASS |
| SOLUSDT | 0.966 | +152.8% | -19.6% | 125 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.775 | +19.8% | -11.4% | 51 | PASS |
| SOLUSDT | 0.037 | +5.7% | -12.4% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla Pivot H3/L3 Breakout + Volume Spike + ATR Filter
Hypothesis: Camarilla H3/L3 levels represent strong intraday support/resistance where breakouts indicate institutional participation.
Volume confirms real money involvement, ATR filter ensures sufficient momentum. Works in bull/bear via breakout logic.
Target: 20-50 trades/year on 4h.
"""

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
    
    # Load 1d data ONCE before loop for HTF pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4 (using H3/L3 for breakout)
    # H3 = close + (high-low)*1.1/2, L3 = close - (high-low)*1.1/2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR filter: ensure sufficient volatility
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > (atr_ma * 0.8)  # Trade when volatility is above 80% of its 50-period MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        atr_ok = atr_filter[i]
        
        # Camarilla breakout conditions
        breakout_long = curr_close > camarilla_h3_aligned[i-1]  # Break above previous period's H3
        breakout_short = curr_close < camarilla_l3_aligned[i-1]  # Break below previous period's L3
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume + ATR filter
            long_entry = breakout_long and vol_spike and atr_ok
            short_entry = breakout_short and vol_spike and atr_ok
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches Camarilla L3 level
            if curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches Camarilla H3 level
            if curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_ATRFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:07
