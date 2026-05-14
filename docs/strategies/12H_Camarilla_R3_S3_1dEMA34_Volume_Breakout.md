# Strategy: 12H_Camarilla_R3_S3_1dEMA34_Volume_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.391 | +10.5% | -7.9% | 92 | FAIL |
| ETHUSDT | 0.091 | +24.1% | -7.5% | 81 | PASS |
| SOLUSDT | -0.005 | +17.0% | -19.6% | 83 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.058 | +6.4% | -4.8% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 (1d) AND price > 1d EMA34 (uptrend) AND volume > 1.8x average.
Short when price breaks below Camarilla S3 (1d) AND price < 1d EMA34 (downtrend) AND volume > 1.8x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA34).
Designed for low trade frequency (~15-25/year) to capture strong breakouts in trending markets while avoiding false signals in ranging conditions.
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
    
    # Load 1d data for Camarilla pivot and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    typical_price = (high_1d + low_1d + close_1d) / 3
    price_range = high_1d - low_1d
    camarilla_pp = typical_price
    camarilla_r3 = camarilla_pp + price_range * 1.1 / 2
    camarilla_s3 = camarilla_pp - price_range * 1.1 / 2
    
    # Calculate EMA34 for 1d trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r3_val and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s3_val and price < ema34_val and vol_current > 1.8 * vol_ma_val):
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

name = "12H_Camarilla_R3_S3_1dEMA34_Volume_Breakout"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:12
