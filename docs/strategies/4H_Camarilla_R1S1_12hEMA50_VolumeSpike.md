# Strategy: 4H_Camarilla_R1S1_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.275 | +29.1% | -3.7% | 158 | PASS |
| ETHUSDT | 0.598 | +41.5% | -6.4% | 134 | PASS |
| SOLUSDT | -0.033 | +18.2% | -12.2% | 108 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.675 | -4.0% | -7.7% | 59 | FAIL |
| ETHUSDT | 0.272 | +8.5% | -6.8% | 51 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 level AND close > 12h EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S1 level AND close < 12h EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Camarilla Pivot point (central level).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 19-50 trades/year per symbol.
Camarilla R1/S1 levels (close ± 1.083 * daily range) provide optimal breakout validation with sufficient frequency.
12h EMA50 offers smooth trend filter with lower lag than longer EMAs. Volume confirmation at 2.0x ensures institutional-grade breakouts.
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
    
    # Load 12h data for EMA50 trend filter and Camarilla levels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 12h OHLC
    range_12h = high_12h - low_12h
    camarilla_r1_12h = close_12h + 1.083 * range_12h   # R1: close + 1.083 * range
    camarilla_s1_12h = close_12h - 1.083 * range_12h   # S1: close - 1.083 * range
    camarilla_pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Align HTF indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND close > 12h EMA50 AND volume spike
            if (price > camarilla_r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND close < 12h EMA50 AND volume spike
            elif (price < camarilla_s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
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

name = "4H_Camarilla_R1S1_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 04:45
