# Strategy: 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Zone

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.092 | +17.3% | -8.8% | 267 | FAIL |
| ETHUSDT | 0.149 | +26.9% | -13.8% | 247 | PASS |
| SOLUSDT | 0.627 | +61.0% | -10.8% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.679 | +31.7% | -7.3% | 91 | PASS |
| SOLUSDT | 0.561 | +13.4% | -7.4% | 78 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Zone"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: current volume > 2.0x 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    # Price range filter: avoid choppy markets (ATR-based)
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr2 = np.maximum(np.absolute(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[tr1[0]], tr2]) if len(tr1) > 0 else np.array([0.0])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Normalize ATR by price to get percentage
    atr_pct = atr / close
    # Only trade when volatility is moderate (not too high, not too low)
    vol_regime = (atr_pct > 0.01) & (atr_pct < 0.05)  # 1% to 5% ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_filter[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R3 + above 1d EMA34 + volume filter + vol regime
            if high[i] > r3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 + below 1d EMA34 + volume filter + vol regime
            elif low[i] < s3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below S3 or below 1d EMA34
            if low[i] < s3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above R3 or above 1d EMA34
            if high[i] > r3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 03:36
