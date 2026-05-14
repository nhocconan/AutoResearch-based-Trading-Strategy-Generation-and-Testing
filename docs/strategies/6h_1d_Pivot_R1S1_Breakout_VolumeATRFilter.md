# Strategy: 6h_1d_Pivot_R1S1_Breakout_VolumeATRFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.009 | +19.8% | -9.8% | 150 | PASS |
| ETHUSDT | 0.513 | +54.7% | -15.4% | 129 | PASS |
| SOLUSDT | 0.846 | +126.2% | -19.8% | 104 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.855 | -13.0% | -16.3% | 60 | FAIL |
| ETHUSDT | 1.335 | +30.6% | -6.3% | 41 | PASS |
| SOLUSDT | -0.254 | +0.4% | -18.9% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Pivot R1/S1 breakout, volume confirmation, and ATR-based volatility filter.
# Uses 1d pivot levels as dynamic support/resistance, enters on breakouts with volume and volatility confirmation.
# Designed to work in both bull and bear markets by filtering for genuine momentum bursts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "6h_1d_Pivot_R1S1_Breakout_VolumeATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Pivot calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot, R1, S1 on 1d timeframe
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        # Volatility filter: require ATR > 0.5 * 50-period ATR average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > (0.5 * atr_ma[i]) if not np.isnan(atr_ma[i]) else True
        
        if position == 0:
            # Long when price breaks above R1 with volume and volatility
            if close[i] > r1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume and volatility
            elif close[i] < s1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal) or volatility drops
            if close[i] < s1_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal) or volatility drops
            if close[i] > r1_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 16:37
