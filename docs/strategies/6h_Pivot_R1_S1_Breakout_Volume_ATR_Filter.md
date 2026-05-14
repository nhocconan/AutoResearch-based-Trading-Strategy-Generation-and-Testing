# Strategy: 6h_Pivot_R1_S1_Breakout_Volume_ATR_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.320 | +11.5% | -5.9% | 517 | FAIL |
| ETHUSDT | 0.125 | +25.5% | -7.3% | 525 | PASS |
| SOLUSDT | 0.222 | +32.6% | -14.2% | 439 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.462 | +10.6% | -4.6% | 161 | PASS |
| SOLUSDT | 1.354 | +20.1% | -6.9% | 156 | PASS |

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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily pivot points (standard floor trader's pivots)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 6h price breaks above R1 with volume confirmation → long
        # 2. 6h price breaks below S1 with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above R1
        if (close[i] > r1_6h[i] and            # 6h price above R1 pivot
            volume_ratio[i] > 1.3 and          # Volume confirmation
            atr_14_6h[i] > 0.005 * close[i]):  # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below S1
        elif (close[i] < s1_6h[i] and          # 6h price below S1 pivot
              volume_ratio[i] > 1.3 and        # Volume confirmation
              atr_14_6h[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-15 10:38
