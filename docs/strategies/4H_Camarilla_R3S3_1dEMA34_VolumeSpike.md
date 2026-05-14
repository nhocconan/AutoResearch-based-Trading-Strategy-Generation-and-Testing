# Strategy: 4H_Camarilla_R3S3_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.318 | +13.5% | -6.4% | 100 | FAIL |
| ETHUSDT | 0.367 | +33.1% | -5.0% | 79 | PASS |
| SOLUSDT | 0.501 | +48.9% | -9.2% | 70 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.851 | +13.5% | -4.9% | 29 | PASS |
| SOLUSDT | -0.982 | -1.9% | -6.0% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 level AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price crosses Camarilla Pivot point (central level).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 19-50 trades/year per symbol.
Camarilla R3/S3 levels (close ± 1.25 * daily range) provide optimal breakout validation with sufficient frequency.
1d EMA34 offers smooth trend filter with lower lag than longer EMAs. Volume confirmation at 2.0x ensures institutional-grade breakouts.
Designed to work in both bull and bear markets by using HTF trend filter and volatility-adjusted entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter and Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    range_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + 1.25 * range_1d   # R3: close + 1.25 * range
    camarilla_s3_1d = close_1d - 1.25 * range_1d   # S3: close - 1.25 * range
    camarilla_pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align HTF indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Ensure warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 1d EMA34 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 1d EMA34 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla Pivot point
            if position == 1 and price < camarilla_pivot_aligned[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_pivot_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 04:43
