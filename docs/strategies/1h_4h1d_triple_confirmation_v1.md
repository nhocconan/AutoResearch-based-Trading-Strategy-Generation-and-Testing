# Strategy: 1h_4h1d_triple_confirmation_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.211 | -11.3% | -17.8% | 558 | FAIL |
| ETHUSDT | -0.259 | +10.1% | -8.5% | 549 | FAIL |
| SOLUSDT | 0.046 | +20.4% | -21.4% | 503 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.147 | +7.5% | -8.1% | 165 | PASS |

## Code
```python
#!/usr/bin/env python3
# 1h_4h1d_triple_confirmation_v1
# Hypothesis: Combine 4h trend (EMA50), 1d momentum (RSI), and 1h breakout with volume filter.
# Long when: 1h price > 1h EMA20, 4h EMA50 rising, 1d RSI > 50, and volume > 1.5x 20-period average.
# Short when: 1h price < 1h EMA20, 4h EMA50 falling, 1d RSI < 50, and volume > 1.5x 20-period average.
# Exit when 1h price crosses 1h EMA20 in opposite direction.
# Uses multiple timeframe confirmation to reduce false signals and whipsaws.
# Target: 20-40 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_triple_confirmation_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # 1d RSI momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h EMA20 for entry/exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below EMA20
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price above EMA20
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: Price above EMA20, 4h EMA50 rising, 1d RSI > 50, volume surge
            if (close[i] > ema_20[i] and 
                ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1] and 
                rsi_1d_aligned[i] > 50 and 
                vol_surge):
                position = 1
                signals[i] = 0.20
            # Short entry: Price below EMA20, 4h EMA50 falling, 1d RSI < 50, volume surge
            elif (close[i] < ema_20[i] and 
                  ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1] and 
                  rsi_1d_aligned[i] < 50 and 
                  vol_surge):
                position = -1
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-08 19:15
