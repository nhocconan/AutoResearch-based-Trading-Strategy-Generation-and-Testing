# Strategy: 12h_Camarilla_Pullback_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.410 | +41.5% | -11.0% | 109 | PASS |
| ETHUSDT | 0.062 | +22.0% | -15.3% | 167 | PASS |
| SOLUSDT | 0.878 | +120.6% | -21.0% | 263 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.767 | +0.1% | -5.0% | 31 | FAIL |
| ETHUSDT | 0.333 | +10.0% | -4.9% | 41 | PASS |
| SOLUSDT | 0.343 | +10.6% | -6.5% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pullback_1dTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    prev_close = np.roll(df_1d['close'], 1)
    prev_high = np.roll(df_1d['high'], 1)
    prev_low = np.roll(df_1d['low'], 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels: H4, L4 (key resistance/support)
    H4 = (prev_high + prev_low) * 1.1 / 2 - (prev_high - prev_low) * 1.1 / 6
    L4 = (prev_high + prev_low) * 1.1 / 2 + (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need 34 for EMA + 1 for roll
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        h4 = H4_aligned[i]
        l4 = L4_aligned[i]
        
        if position == 0:
            # Enter long: Pullback to L4 in uptrend (price > EMA34)
            if close[i] <= l4 and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Pullback to H4 in downtrend (price < EMA34)
            elif close[i] >= h4 and close[i] < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below EMA34 (trend change) or reaches H4 (target)
            if close[i] < ema_1d or close[i] >= h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above EMA34 (trend change) or reaches L4 (target)
            if close[i] > ema_1d or close[i] <= l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 03:38
