# Strategy: 4H_4H_Camarilla_R1S1_Breakout_TrendFilter_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.069 | +18.6% | -11.4% | 457 | FAIL |
| ETHUSDT | 0.191 | +28.4% | -9.1% | 424 | PASS |
| SOLUSDT | -0.068 | +13.9% | -21.1% | 361 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.510 | +12.3% | -6.8% | 166 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4H_4H_Camarilla_R1S1_Breakout_TrendFilter_Volume
Hypothesis: Uses Camarilla pivot levels from 4h data (R1/S1) for breakout entries, confirmed by
4h EMA50 trend and volume spikes. Designed for low trade frequency by requiring confluence of
price breaking key pivot levels, trend alignment, and volume confirmation. Works in bull and bear
markets by following intermediate-term trend on the same timeframe (4h).
"""

name = "4H_4H_Camarilla_R1S1_Breakout_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla Pivot Levels from 4h data ---
    # Calculate pivot points using previous 4h bar's OHLC
    # For each bar, we need the previous bar's OHLC to calculate today's pivots
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar will have NaN due to roll, handled later
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    
    # --- 4h EMA50 Trend Filter ---
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and pivot calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars after roll)
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above R1 with volume, above EMA50
            if (close[i] > R1[i] and 
                volume_spike and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below EMA50
            elif (close[i] < S1[i] and 
                  volume_spike and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below S1 (reversal signal)
                if close[i] < S1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 (reversal signal)
                if close[i] > R1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 06:38
