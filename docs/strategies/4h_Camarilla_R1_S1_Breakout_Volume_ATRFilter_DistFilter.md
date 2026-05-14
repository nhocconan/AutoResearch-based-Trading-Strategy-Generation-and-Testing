# Strategy: 4h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_DistFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.289 | +32.8% | -8.1% | 321 | PASS |
| ETHUSDT | 0.106 | +24.9% | -8.8% | 305 | PASS |
| SOLUSDT | 0.643 | +75.3% | -21.5% | 261 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.065 | -3.1% | -9.9% | 134 | FAIL |
| ETHUSDT | 0.262 | +9.2% | -10.0% | 114 | PASS |
| SOLUSDT | 0.420 | +11.5% | -10.4% | 93 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily OHLC for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1-S1)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = close_1d + range_hl * 1.1 / 12
    s1 = close_1d - range_hl * 1.1 / 12
    
    # === ATR for volatility filter (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF data to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    atr_1d_avg_4h = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Price distance from pivot (avoid chop) ===
    mid_pivot = (r1_4h + s1_4h) / 2
    dist_from_pivot = np.abs(close - mid_pivot)
    avg_dist = pd.Series(dist_from_pivot).rolling(window=50, min_periods=50).mean().values
    too_close = dist_from_pivot < (0.5 * avg_dist)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(atr_1d_avg_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(too_close[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_4h[i]
        s1_level = s1_4h[i]
        atr_avg = atr_1d_avg_4h[i]
        vol_spike = volume_spike[i]
        too_close_to_pivot = too_close[i]
        
        # === EXIT LOGIC: Exit when price moves against position or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below S1 or volatility drops significantly
            if price < s1_level or atr_avg < (atr_1d_avg_4h[i-1] * 0.7 if i > 0 else atr_avg):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 or volatility drops significantly
            if price > r1_level or atr_avg < (atr_1d_avg_4h[i-1] * 0.7 if i > 0 else atr_avg):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, sufficient volatility, and not too close to pivot
            if price > r1_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, sufficient volatility, and not too close to pivot
            elif price < s1_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_DistFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 15:49
