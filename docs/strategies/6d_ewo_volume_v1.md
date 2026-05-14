# Strategy: 6d_ewo_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.564 | +0.2% | -10.9% | 161 | FAIL |
| ETHUSDT | 0.343 | +40.0% | -15.9% | 157 | PASS |
| SOLUSDT | 0.560 | +71.5% | -15.5% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.051 | +6.1% | -7.8% | 52 | PASS |
| SOLUSDT | -0.087 | +3.9% | -12.6% | 45 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6d_ewo_volume_v1
# Hypothesis: Uses Elder's Force Index (EFI) with volume confirmation on 6h timeframe.
# EFI measures buying/selling pressure by combining price change and volume.
# Long when EFI crosses above zero with volume > 1.5x average; short when EFI crosses below zero.
# Designed to work in both bull and bear markets by capturing momentum shifts.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6d_ewo_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. Elder's Force Index (EFI) - 13 period EMA of price change * volume
    price_change = np.diff(close, prepend=close[0])
    efi_raw = price_change * volume
    
    # EMA of EFI
    efi = np.zeros(n)
    efi[0] = efi_raw[0]
    alpha = 2.0 / (13 + 1)  # 13-period EMA
    for i in range(1, n):
        efi[i] = alpha * efi_raw[i] + (1 - alpha) * efi[i-1]
    
    # 2. Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(efi[i]) or np.isnan(efi[i-1]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: EFI crosses below zero
            if efi[i] < 0 and efi[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EFI crosses above zero
            if efi[i] > 0 and efi[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: EFI crosses above zero with volume confirmation
            if efi[i] > 0 and efi[i-1] <= 0 and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: EFI crosses below zero with volume confirmation
            elif efi[i] < 0 and efi[i-1] >= 0 and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 06:55
