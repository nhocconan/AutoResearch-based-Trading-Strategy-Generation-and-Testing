# Strategy: 1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Filtered

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.144 | +26.3% | -6.0% | 265 | PASS |
| ETHUSDT | -0.368 | +4.5% | -15.8% | 279 | FAIL |
| SOLUSDT | 0.869 | +106.5% | -18.4% | 253 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.080 | +6.7% | -4.0% | 91 | PASS |
| SOLUSDT | 0.353 | +10.6% | -9.3% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Filtered"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Camarilla pivot levels (R1, S1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = close_4h + (range_4h * 1.0833)
    s1_4h = close_4h - (range_4h * 1.0833)
    
    # Align levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (1h volume > 1.8x 24-period average)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.8 * volume_ma24
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        if np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above EMA34, volume confirmation, session
            if close[i] > r1_4h_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S1, below EMA34, volume confirmation, session
            elif close[i] < s1_4h_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below EMA34
            if close[i] < s1_4h_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Close above R1 or above EMA34
            if close[i] > r1_4h_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-05-11 21:27
