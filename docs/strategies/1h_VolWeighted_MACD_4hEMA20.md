# Strategy: 1h_VolWeighted_MACD_4hEMA20

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.246 | +14.5% | -6.9% | 554 | DISCARD |
| ETHUSDT | 0.632 | +44.7% | -5.5% | 506 | KEEP |
| SOLUSDT | 0.001 | +18.9% | -15.8% | 478 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.144 | +7.3% | -5.4% | 173 | KEEP |
| SOLUSDT | -0.222 | +3.6% | -5.2% | 154 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
1h Volume-Weighted MACD with 4h Trend Filter
Long: MACD histogram crosses above zero AND price above 4h EMA(20) AND volume > 1.5x 1h volume SMA(20)
Short: MACD histogram crosses below zero AND price below 4h EMA(20) AND volume > 1.5x 1h volume SMA(20)
Exit: MACD histogram crosses back to opposite side of zero
Uses 4h EMA for trend direction, MACD for momentum, volume for confirmation
Target: 15-30 trades/year per symbol (60-120 total over 4 years)
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA(20) for trend direction
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate MACD (12,26,9) on 1h data
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - signal_line
    
    # Calculate 1h volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 26 + 9)  # MACD needs 26+9 bars for signal line
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(macd_hist[i-1]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        ema_4h_val = ema_20_4h_aligned[i]
        macd_h = macd_hist[i]
        macd_h_prev = macd_hist[i-1]
        
        if position == 0:
            # Long: MACD hist crosses above zero + price above 4h EMA + volume > 1.5x SMA
            if macd_h_prev <= 0 and macd_h > 0 and price > ema_4h_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.20
                position = 1
            # Short: MACD hist crosses below zero + price below 4h EMA + volume > 1.5x SMA
            elif macd_h_prev >= 0 and macd_h < 0 and price < ema_4h_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: MACD hist crosses below zero
            if macd_h_prev >= 0 and macd_h < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: MACD hist crosses above zero
            if macd_h_prev <= 0 and macd_h > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolWeighted_MACD_4hEMA20"
timeframe = "1h"
leverage = 1.0
```

## Last Updated
2026-04-17 23:19
