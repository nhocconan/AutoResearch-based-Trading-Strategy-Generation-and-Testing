# Strategy: 4H_Camarilla_R3S3_1wEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.607 | +33.4% | -2.6% | 179 | PASS |
| ETHUSDT | 0.723 | +39.2% | -3.3% | 158 | PASS |
| SOLUSDT | -0.389 | +7.8% | -14.1% | 108 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.755 | -4.6% | -5.0% | 62 | FAIL |
| ETHUSDT | 0.200 | +7.5% | -5.2% | 60 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout + 1w EMA50 trend filter + volume spike.
Long when price breaks above Camarilla R3 level AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 level AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla Pivot level (midpoint) or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, while 1w EMA50 filters for higher-timeframe trend alignment.
Volume spike confirms institutional participation. ATR stoploss manages risk. Works in both bull and bear markets by avoiding counter-trend breakouts.
"""

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
    
    # Load 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (based on previous 4h bar) - using typical formula
    # Pivot = (H+L+C)/3
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    # We use rolling window of 1 for previous bar values
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 2.0
    s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 2.0
    
    # Align 4h Camarilla levels to 4h timeframe (shift by 1 to use previous bar's levels)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND close > 1w EMA50 AND volume spike
            if (price > r3_4h_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND close < 1w EMA50 AND volume spike
            elif (price < s3_4h_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot or ATR stoploss
                if price <= pivot_4h_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot or ATR stoploss
                if price >= pivot_4h_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 04:05
