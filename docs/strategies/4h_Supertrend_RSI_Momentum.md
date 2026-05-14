# Strategy: 4h_Supertrend_RSI_Momentum

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.130 | +11.1% | -12.9% | 270 | FAIL |
| ETHUSDT | 0.215 | +32.9% | -16.2% | 257 | PASS |
| SOLUSDT | 1.085 | +204.8% | -30.7% | 225 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.570 | +16.2% | -9.7% | 86 | PASS |
| SOLUSDT | 0.360 | +12.3% | -13.2% | 82 | PASS |

## Code
```python
# 1:100
#!/usr/bin/env python3
"""
4h_Supertrend_RSI_Momentum
Hypothesis: Supertrend identifies trend direction, RSI measures momentum strength, and volume filters false breakouts. Works in bull/bear via trend filter and avoids chop via momentum threshold. Designed for low trade frequency to minimize fee drag.
"""

name = "4h_Supertrend_RSI_Momentum"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR(10) for Supertrend
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    supertrend = np.full(n, np.nan)
    dir = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    dir[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            dir[i] = 1
        else:
            dir[i] = -1
        
        if dir[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Supertrend uptrend, RSI > 50 (bullish momentum), volume confirmation, price above 12h EMA50
            if (dir[i] == 1 and 
                rsi[i] > 50 and 
                volume_filter[i] and 
                close[i] > trend_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend, RSI < 50 (bearish momentum), volume confirmation, price below 12h EMA50
            elif (dir[i] == -1 and 
                  rsi[i] < 50 and 
                  volume_filter[i] and 
                  close[i] < trend_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Supertrend turns down OR RSI < 40 (loss of momentum)
            if (dir[i] == -1 or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend turns up OR RSI > 60 (loss of bearish momentum)
            if (dir[i] == 1 or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 08:15
