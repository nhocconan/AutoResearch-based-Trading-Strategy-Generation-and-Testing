# Strategy: 6H_Camarilla_R4_S4_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.279 | +34.7% | -14.7% | 169 | PASS |
| ETHUSDT | 0.105 | +24.6% | -15.8% | 159 | PASS |
| SOLUSDT | 1.195 | +221.0% | -19.5% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.443 | -9.3% | -12.0% | 60 | FAIL |
| ETHUSDT | 1.128 | +28.1% | -7.1% | 49 | PASS |
| SOLUSDT | 0.723 | +20.1% | -9.9% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation.
Long when price breaks above R4 and close > 1d EMA34 (uptrend) with volume > 1.5x average.
Short when price breaks below S4 and close < 1d EMA34 (downtrend) with volume > 1.5x average.
Uses 6h timeframe to target 50-150 total trades over 4 years. Camarilla levels from 1d provide
intraday support/resistance structure. Volume confirmation ensures breakout conviction.
Trend filter prevents counter-trend trades. Works in both bull and bear markets by aligning
with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1 / 2
    # S4 = PP - (H - L) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s4 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R4 AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (price > r4_val and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4 AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (price < s4_val and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S4 OR price breaks below 1d EMA34 (trend reversal)
                if price < s4_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R4 OR price breaks above 1d EMA34 (trend reversal)
                if price > r4_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4_S4_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:57
