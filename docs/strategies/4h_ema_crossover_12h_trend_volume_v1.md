# Strategy: 4h_ema_crossover_12h_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.640 | -0.5% | -16.4% | 102 | FAIL |
| ETHUSDT | 0.050 | +22.0% | -15.6% | 103 | PASS |
| SOLUSDT | 0.906 | +124.0% | -15.5% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.543 | +14.0% | -9.9% | 37 | PASS |
| SOLUSDT | -0.352 | +0.5% | -12.5% | 33 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_ema_crossover_12h_trend_volume_v1
Hypothesis: EMA crossover (9/21) on 4h timeframe filtered by 12-hour EMA50 trend and volume confirmation.
In long: fast EMA crosses above slow EMA with volume > 20-period average and price above 12h EMA50.
In short: fast EMA crosses below slow EMA with volume > 20-period average and price below 12h EMA50.
Designed for 20-30 trades/year on 4h timeframe with clear trend-following logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_crossover_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # EMA crossover (9/21) on 4h
    ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if data not available
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # EMA crossover conditions
        bullish_cross = ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]
        bearish_cross = ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]
        
        # 12h trend filter
        above_12h_ema50 = close[i] > ema50_12h_aligned[i]
        below_12h_ema50 = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish crossover or trend turns bearish
            if bearish_cross or below_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish crossover or trend turns bullish
            if bullish_cross or above_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish EMA crossover with volume confirmation and bullish trend
            if bullish_cross and vol_confirmed and above_12h_ema50:
                position = 1
                signals[i] = 0.25
            # Short: bearish EMA crossover with volume confirmation and bearish trend
            elif bearish_cross and vol_confirmed and below_12h_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 20:45
