# Strategy: 4h_Vortex_Volume_Spike_1dTrend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.557 | +47.7% | -6.3% | 186 | PASS |
| ETHUSDT | 0.529 | +51.4% | -12.3% | 183 | PASS |
| SOLUSDT | 0.413 | +55.1% | -21.9% | 161 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.271 | -5.6% | -6.2% | 67 | FAIL |
| ETHUSDT | 0.763 | +17.2% | -12.2% | 59 | PASS |
| SOLUSDT | -0.217 | +2.2% | -7.7% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Vortex_Volume_Spike_1dTrend_v1
Hypothesis: On 4h timeframe, use Vortex Indicator to identify trend direction and strength, 
filtered by daily EMA trend and volume spikes to avoid false signals. 
Long when VI+ > VI- and price above daily EMA with volume spike. 
Short when VI- > VI+ and price below daily EMA with volume spike. 
Vortex helps distinguish between trending and ranging markets, reducing whipsaws in chop.
Works in both bull and bear markets by requiring alignment with daily trend.
"""
name = "4h_Vortex_Volume_Spike_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (period=14)
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(34, 20, 14)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades to reduce frequency (4h timeframe)
            if bars_since_entry < 8:
                continue
                
            # Long: VI+ > VI- (bullish vortex) + price above EMA34 + volume filter
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: VI- > VI+ (bearish vortex) + price below EMA34 + volume filter
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: Vortex crossover in opposite direction
            if position == 1:
                if vi_minus[i] > vi_plus[i]:  # Bearish crossover
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if vi_plus[i] > vi_minus[i]:  # Bullish crossover
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 07:19
