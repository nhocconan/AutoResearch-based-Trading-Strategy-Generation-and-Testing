# Strategy: 4h_Camarilla_R1S1_Breakout_Volume_Spike_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.557 | +39.4% | -5.4% | 283 | PASS |
| ETHUSDT | 0.008 | +21.0% | -6.8% | 263 | PASS |
| SOLUSDT | 0.516 | +54.2% | -16.8% | 229 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.406 | -3.4% | -5.8% | 108 | FAIL |
| ETHUSDT | 0.767 | +14.3% | -9.5% | 97 | PASS |
| SOLUSDT | 0.740 | +13.9% | -5.4% | 84 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Spike_TrendFilter
Hypothesis: Camarilla R1/S1 breakouts on 4h with volume spike and 1d EMA trend filter.
Buy when price breaks above R1 with volume spike and uptrend (price > EMA34).
Sell when price breaks below S1 with volume spike and downtrend (price < EMA34).
Designed for low trade frequency (20-50/year) to avoid fee drag while capturing
significant momentum moves in both bull and bear markets via trend alignment.
"""

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous day's data)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align to 4h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1_val and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1_val and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below R1 OR trend turns down
            if price < r1_val or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above S1 OR trend turns up
            if price > s1_val or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Spike_TrendFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 02:53
