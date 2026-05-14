# Strategy: 12h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.028 | +18.2% | -11.0% | 185 | FAIL |
| ETHUSDT | 0.362 | +42.1% | -16.4% | 163 | PASS |
| SOLUSDT | 0.367 | +51.9% | -29.6% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.592 | +15.6% | -8.5% | 57 | PASS |
| SOLUSDT | -0.850 | -9.2% | -21.4% | 52 | FAIL |

## Code
```python
# This strategy implements a 12h pivot (R1/S1) breakout system with volume confirmation and ATR-based volatility filtering.
# It aims to capture breakouts from key daily-derived pivot levels while avoiding low-volatility false breakouts.
# The strategy works in both bull and bear markets: in bull markets, breakouts tend to continue; in bear markets,
# price often reverts to the mean at S1/R1 levels during ranging conditions, allowing for mean-reversion exits.
# Target: 20-150 total trades over 4 years (5-37/year) to minimize fee drag and improve generalization.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous daily bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # Calculate daily ATR (14) for volatility filter
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily R1/S1 and ATR to 12h (wait for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        atr_val = atr_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume confirmation and sufficient volatility
            if close_val > R1_val and vol_filter and (atr_val > 0):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and sufficient volatility
            elif close_val < S1_val and vol_filter and (atr_val > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 or ATR drops too low (low volatility)
            if close_val < S1_val or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or ATR drops too low (low volatility)
            if close_val > R1_val or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-18 23:05
