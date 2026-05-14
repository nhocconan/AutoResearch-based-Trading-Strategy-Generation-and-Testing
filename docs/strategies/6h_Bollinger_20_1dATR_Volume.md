# Strategy: 6h_Bollinger_20_1dATR_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.092 | +24.2% | -9.1% | 98 | PASS |
| ETHUSDT | 0.398 | +46.0% | -12.7% | 90 | PASS |
| SOLUSDT | 1.041 | +175.2% | -29.3% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.482 | -8.0% | -10.2% | 38 | FAIL |
| ETHUSDT | 0.170 | +8.0% | -7.9% | 35 | PASS |
| SOLUSDT | -0.694 | -7.7% | -18.7% | 33 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-day Bollinger Bands (20, 2.0)
    sma_20_1d = np.full(len(close_1d), np.nan)
    std_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
            std_20_1d[i] = np.std(close_1d[i-19:i+1])
    
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    
    # Align 1d indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Bollinger Band with volume
            if price > upper_bb_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Bollinger Band with volume
            elif price < lower_bb_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower Bollinger Band or volatility spike (potential reversal)
            if price < lower_bb_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper Bollinger Band or volatility spike (potential reversal)
            if price > upper_bb_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Bollinger_20_1dATR_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 13:27
