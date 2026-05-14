# Strategy: 12h_Camarilla_R1_S1_Breakout_1dEMA200_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.087 | +16.8% | -9.1% | 36 | FAIL |
| ETHUSDT | 0.095 | +24.3% | -17.4% | 37 | PASS |
| SOLUSDT | 0.900 | +118.9% | -21.0% | 32 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.388 | +11.8% | -11.5% | 13 | PASS |
| SOLUSDT | 0.295 | +10.2% | -9.2% | 13 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dEMA200_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # 1. Load 1d data ONCE for trend and pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # 2. 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 3. Calculate daily high/low/close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4. Camarilla levels: R1, S1 (inner levels for tighter entries)
    hl_range = high_1d - low_1d
    r1 = close_1d + hl_range * 1.09
    s1 = close_1d - hl_range * 1.09
    
    # 5. Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 6. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 7. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 + above 1d EMA200 + volume spike
            if breakout_long and price_above_ema200 and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S1 + below 1d EMA200 + volume spike
            elif breakout_short and price_below_ema200 and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S1 OR trend reverses
                if close[i] < s1_aligned[i] or close[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R1 OR trend reverses
                if close[i] > r1_aligned[i] or close[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals
```

## Last Updated
2026-05-11 09:44
