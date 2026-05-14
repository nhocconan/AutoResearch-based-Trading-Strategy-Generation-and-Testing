# Strategy: 12h_KAMA_Trend_1dRSI_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.228 | +7.0% | -18.9% | 43 | FAIL |
| ETHUSDT | 0.130 | +26.3% | -16.8% | 41 | PASS |
| SOLUSDT | 1.031 | +187.8% | -20.0% | 39 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.192 | +8.5% | -11.4% | 15 | PASS |
| SOLUSDT | -0.314 | -2.4% | -20.3% | 17 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h KAMA Trend + 1d RSI + Volume Spike
Long: KAMA rising + RSI(1d) > 55 + volume > 1.5x 12h volume SMA(20)
Short: KAMA falling + RSI(1d) < 45 + volume > 1.5x 12h volume SMA(20)
Exit: Opposite KAMA direction or RSI crosses 50
Uses Kaufman Adaptive Moving Average for trend, filtered by daily momentum and volume.
Designed to work in trending markets with confirmation from higher timeframe momentum.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values  # neutral when undefined
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate KAMA(10, 2, 30) for trend
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i == 0:
            er[i] = 0
        else:
            er[i] = change[i] / (volatility[:i+1].sum() + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h volume SMA(20)
    vol_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 30)  # need volume SMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(vol_sma_12h[i]) or
            np.isnan(kama[i]) or np.isnan(kama[i-1])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_12h[i]
        rsi_val = rsi_14_1d_aligned[i]
        kama_val = kama[i]
        kama_prev = kama[i-1]
        
        if position == 0:
            # Long: KAMA rising + RSI > 55 + volume spike
            if kama_val > kama_prev and rsi_val > 55 and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 45 + volume spike
            elif kama_val < kama_prev and rsi_val < 45 and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling or RSI < 50
            if kama_val < kama_prev or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising or RSI > 50
            if kama_val > kama_prev or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_1dRSI_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 23:39
