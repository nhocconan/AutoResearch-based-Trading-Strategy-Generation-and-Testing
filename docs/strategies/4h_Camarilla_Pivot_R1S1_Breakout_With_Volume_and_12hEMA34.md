# Strategy: 4h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_12hEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.251 | +30.7% | -11.0% | 276 | PASS |
| ETHUSDT | 0.068 | +23.1% | -11.8% | 267 | PASS |
| SOLUSDT | 0.601 | +69.0% | -21.4% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.092 | -2.6% | -7.7% | 113 | FAIL |
| ETHUSDT | 0.903 | +18.0% | -4.8% | 101 | PASS |
| SOLUSDT | 0.649 | +14.2% | -10.2% | 77 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_12hEMA34
Hypothesis: Buy when price breaks above Camarilla R1 with volume spike and above 12h EMA34; short when breaks below S1 with volume spike and below 12h EMA34. Camarilla pivots provide precise support/resistance levels based on prior day's range, effective in both trending and ranging markets. Volume confirms institutional participation, and 12h EMA34 ensures alignment with medium-term trend. Designed for low trade frequency to minimize fee drag while capturing high-probability breakouts.
"""

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
    
    # Camarilla pivot levels from previous day
    # Need daily high, low, close
    df_1d = get_htf_data(prices, '1d')
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rang = phigh - plow
    r1 = pclose + rang * 1.1 / 12
    s1 = pclose - rang * 1.1 / 12
    
    # Align to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need volume MA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_spike = volume_spike[i]
        ema_12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: price > R1 with volume spike and above 12h EMA34
            if price > r1_val and vol_spike and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and below 12h EMA34
            elif price < s1_val and vol_spike and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 or below 12h EMA34
            if price < s1_val or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 or above 12h EMA34
            if price > r1_val or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_12hEMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 02:30
