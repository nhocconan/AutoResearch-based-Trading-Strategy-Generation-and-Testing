# Strategy: 12H_Camarilla_R3_S3_1dHMA21_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.013 | +19.9% | -7.4% | 107 | FAIL |
| ETHUSDT | 0.214 | +30.8% | -9.2% | 92 | PASS |
| SOLUSDT | 0.207 | +32.6% | -22.4% | 81 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.370 | +11.1% | -5.9% | 34 | PASS |
| SOLUSDT | -1.146 | -10.6% | -17.8% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d HMA trend filter and volume confirmation.
Long when price breaks above R3 and close > 1d HMA21 (uptrend) with volume > 1.8x average.
Short when price breaks below S3 and close < 1d HMA21 (downtrend) with volume > 1.8x average.
Uses 12h timeframe targeting 50-150 total trades over 4 years. HMA provides smoother trend
than EMA, reducing whipsaw. Volume confirmation ensures breakout conviction. Works in both
bull and bear markets by aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw = 2 * wma2 - wma1
    hma_val = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
    return hma_val.values

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
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Load 1d data for HMA21 trend filter - ONCE before loop
    hma21_1d = hma(close_1d, 21)
    
    # Align HTF indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    hma21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma21_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(hma21_1d_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        hma21_val = hma21_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 1d HMA21 (uptrend) AND volume confirmation
            if (price > r3_val and price > hma21_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND price < 1d HMA21 (downtrend) AND volume confirmation
            elif (price < s3_val and price < hma21_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR price breaks below 1d HMA21 (trend reversal)
                if price < s3_val or price < hma21_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR price breaks above 1d HMA21 (trend reversal)
                if price > r3_val or price > hma21_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3_S3_1dHMA21_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 02:06
