# Strategy: 6h_DailyPivot_R2_S2_Breakout_VolVol

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.040 | +18.8% | -17.5% | 270 | FAIL |
| ETHUSDT | 0.004 | +19.6% | -14.3% | 257 | PASS |
| SOLUSDT | 0.674 | +85.5% | -22.2% | 250 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.399 | +11.0% | -8.0% | 87 | PASS |
| SOLUSDT | 0.184 | +8.1% | -8.1% | 85 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot (using previous day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align daily pivots to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation (20-period MA on 6h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # volume MA20 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.2x 20-period average
        volume_filter = volume[i] > (1.2 * volume_ma20[i])
        # Volatility filter: ATR > 0.5 * 20-period ATR average (avoid low volatility chop)
        atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr[i] > (0.5 * atr_ma20[i]) if not np.isnan(atr_ma20[i]) else False
        
        if position == 0:
            # Long: break above R2 with volume and volatility
            if close[i] > r2_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume and volatility
            elif close[i] < s2_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below R1 or volatility drops
            if close[i] < r1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above S1 or volatility drops
            if close[i] > s1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_R2_S2_Breakout_VolVol"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 08:39
