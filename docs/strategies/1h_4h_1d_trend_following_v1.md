# Strategy: 1h_4h_1d_trend_following_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.371 | +41.1% | -10.9% | 168 | PASS |
| ETHUSDT | 0.087 | +23.1% | -12.0% | 194 | PASS |
| SOLUSDT | 0.837 | +138.9% | -25.2% | 207 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.549 | -0.3% | -5.6% | 71 | FAIL |
| ETHUSDT | 0.768 | +19.6% | -7.4% | 55 | PASS |
| SOLUSDT | 0.586 | +17.6% | -10.2% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_following_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA200 on 4h close
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    # Align to 1h timeframe
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Align to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 1h volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if EMA data not available
        if np.isnan(ema200_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA200
            if close[i] < ema200_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA200
            if close[i] > ema200_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above both 4h EMA200 and 1d EMA50 with volume confirmation
            if close[i] > ema200_4h_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: price below both 4h EMA200 and 1d EMA50 with volume confirmation
            elif close[i] < ema200_4h_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-08 09:11
