# Strategy: 4H_Camarilla_R1_S1_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.499 | +40.2% | -5.7% | 247 | PASS |
| ETHUSDT | 0.442 | +42.2% | -9.2% | 232 | PASS |
| SOLUSDT | 0.714 | +80.9% | -17.2% | 197 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.426 | -4.6% | -5.4% | 98 | FAIL |
| ETHUSDT | 0.991 | +19.2% | -6.4% | 82 | PASS |
| SOLUSDT | 0.584 | +13.3% | -5.9% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above R1 (1d Camarilla) AND price > 1d EMA34 (uptrend) AND volume > 2x average.
Short when price breaks below S1 (1d Camarilla) AND price < 1d EMA34 (downtrend) AND volume > 2x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA34).
Uses 4h timeframe to target ~25-50 trades/year, avoiding fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 1d EMA34 for breakout entries.
"""

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
    
    # Load 1d data for EMA34 trend filter and Camarilla pivot levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 for 1d trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r1_val and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s1_val and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR price breaks below 1d EMA34 (trend reversal)
                if price <= pp_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR price breaks above 1d EMA34 (trend reversal)
                if price >= pp_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:25
