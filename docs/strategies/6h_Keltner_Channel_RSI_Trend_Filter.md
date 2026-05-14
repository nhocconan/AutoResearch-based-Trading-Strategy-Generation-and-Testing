# Strategy: 6h_Keltner_Channel_RSI_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.453 | -5.4% | -19.1% | 39 | FAIL |
| ETHUSDT | 0.048 | +20.6% | -16.5% | 32 | PASS |
| SOLUSDT | 0.813 | +143.6% | -31.7% | 37 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.199 | +8.8% | -13.6% | 13 | PASS |
| SOLUSDT | -0.673 | -10.0% | -22.4% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_Keltner_Channel_RSI_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily data for Keltner Channel and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Keltner Channel components on daily
    # ATR(10)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA(20) as middle line
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    keltner_upper_1d = ema20_1d + (2.0 * atr10_1d)
    keltner_lower_1d = ema20_1d - (2.0 * atr10_1d)
    
    # RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align all to 6h
    keltner_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike on 6h: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_1d_aligned[i]) or np.isnan(keltner_lower_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner + RSI > 50 + volume spike
            if (close[i] > keltner_upper_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + RSI < 50 + volume spike
            elif (close[i] < keltner_lower_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Keltner or RSI < 40
            if close[i] < keltner_lower_1d_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Keltner or RSI > 60
            if close[i] > keltner_upper_1d_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 01:09
