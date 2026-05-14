# Strategy: 4h_CamarillaR1S1_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.440 | +50.7% | -12.9% | 100 | PASS |
| ETHUSDT | 0.058 | +19.5% | -20.3% | 110 | PASS |
| SOLUSDT | 0.750 | +142.3% | -32.6% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.417 | -0.2% | -7.6% | 44 | FAIL |
| ETHUSDT | 0.816 | +24.0% | -9.9% | 32 | PASS |
| SOLUSDT | 0.623 | +20.5% | -10.4% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1/S1) with 1d EMA34 trend filter and volume confirmation.
In uptrend (price > EMA34), buy breakouts above 1d Camarilla R1; in downtrend (price < EMA34), sell breakdowns below 1d Camarilla S1.
Uses 1d for structure (proven effective) and volume to confirm breakout strength. Targets ~25-40 trades/year (100-160 total over 4 years) to avoid fee drag.
Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = Close + (High - Low) * 1.1/12, S1 = Close - (High - Low) * 1.1/12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + rng * 1.1 / 12
    camarilla_s1 = close_1d - rng * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above 1d Camarilla R1 + uptrend + volume spike
            if (price_close > camarilla_r1_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 1d Camarilla S1 + downtrend + volume spike
            elif (price_close < camarilla_s1_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA34 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_CamarillaR1S1_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-21 19:13
