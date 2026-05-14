# Strategy: 12h_PivotBreakout_VolumeSurge_EMA34Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.246 | +14.4% | -7.0% | 85 | FAIL |
| ETHUSDT | 0.242 | +30.1% | -5.2% | 75 | PASS |
| SOLUSDT | 0.384 | +46.1% | -18.2% | 72 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.100 | +6.9% | -3.1% | 31 | PASS |
| SOLUSDT | -0.774 | -2.2% | -11.1% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily data for pivot points and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe (primary)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 12h ATR(14) for volatility filter and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20  # Strong volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema_34_12h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume surge, above daily EMA34
            if (close[i] > s1_12h[i] and vol_surge[i] and close[i] > ema_34_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume surge, below daily EMA34
            elif (close[i] < r1_12h[i] and vol_surge[i] and close[i] < ema_34_12h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite level or volatility drops significantly
            if position == 1:
                if close[i] < pivot_12h[i] or atr[i] < 0.3 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_12h[i] or atr[i] < 0.3 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_PivotBreakout_VolumeSurge_EMA34Trend_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-22 05:14
