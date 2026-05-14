# Strategy: 6H_Bollinger_Breakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.308 | +34.3% | -8.2% | 109 | PASS |
| ETHUSDT | 0.546 | +53.3% | -9.6% | 99 | PASS |
| SOLUSDT | 0.694 | +90.0% | -21.2% | 79 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.357 | -6.9% | -10.6% | 45 | FAIL |
| ETHUSDT | 0.734 | +17.6% | -5.9% | 34 | PASS |
| SOLUSDT | -0.291 | +0.5% | -13.9% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band breakout with 12h trend filter and volume confirmation.
Long when price breaks above upper BB (20,2) AND price > 12h EMA50 (uptrend) AND volume > 2x average.
Short when price breaks below lower BB (20,2) AND price < 12h EMA50 (downtrend) AND volume > 2x average.
Exit when price reverts to middle BB or trend reverses (price crosses 12h EMA50).
Uses 6h timeframe to target ~15-30 trades/year, avoiding fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 12h EMA50 for breakout entries.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 for 12h trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Bollinger Bands on 6h timeframe
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        bb_middle_val = bb_middle[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper BB AND price > 12h EMA50 (uptrend) AND volume spike
            if (price > bb_upper_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND price < 12h EMA50 (downtrend) AND volume spike
            elif (price < bb_lower_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle BB OR price breaks below 12h EMA50 (trend reversal)
                if price <= bb_middle_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle BB OR price breaks above 12h EMA50 (trend reversal)
                if price >= bb_middle_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Bollinger_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:23
