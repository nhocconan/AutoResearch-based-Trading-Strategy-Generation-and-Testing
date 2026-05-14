# Strategy: 6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.311 | +41.3% | -18.5% | 56 | PASS |
| ETHUSDT | 0.309 | +43.6% | -17.8% | 56 | PASS |
| SOLUSDT | 0.875 | +194.5% | -39.2% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.548 | -3.1% | -12.0% | 23 | FAIL |
| ETHUSDT | 0.936 | +31.3% | -9.1% | 19 | PASS |
| SOLUSDT | -0.109 | +0.2% | -22.4% | 16 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:  # Need sufficient data for daily calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # P = (H + L + C) / 3
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    
    # Calculate for each day using previous day's data
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    range_hl = prev_high - prev_low
    
    r4 = prev_close + range_hl * 1.1
    s4 = prev_close - range_hl * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema34_1d_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R4 with volume, trend up
            if (close[i] > r4_6h[i] and 
                close[i] > ema34_1d_6h[i] and  # Uptrend filter
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Break below S4 with volume, trend down
            elif (close[i] < s4_6h[i] and 
                  close[i] < ema34_1d_6h[i] and  # Downtrend filter
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # EXIT LONG: Trend breaks down
            if close[i] < ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Trend breaks up
            if close[i] > ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-12 06:05
