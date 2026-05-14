# Strategy: 4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_PPExit_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.253 | +30.3% | -8.2% | 200 | PASS |
| ETHUSDT | 0.430 | +41.0% | -9.0% | 191 | PASS |
| SOLUSDT | 0.862 | +98.1% | -19.5% | 151 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.956 | -8.0% | -8.8% | 82 | FAIL |
| ETHUSDT | 0.692 | +14.4% | -6.2% | 64 | PASS |
| SOLUSDT | -0.223 | +3.0% | -8.6% | 56 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike.
Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla pivot (PP) OR ATR trailing stop (2.0*ATR from extreme).
Uses 12h HTF for trend alignment (more responsive than 1d) and Camarilla levels from daily.
Target: ~20-30 trades/year on 4h timeframe with discrete sizing 0.25.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    # R3 = PP + 1.1 * Range / 2, S3 = PP - 1.1 * Range / 2
    r3 = pp + 1.1 * rng / 2.0
    s3 = pp - 1.1 * rng / 2.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 4h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 4h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # vol_ma20, ema_50_12h
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_12h_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pp_val = pp_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 12h EMA50 AND volume spike
            if price > r3_val and price > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below S3 AND price < 12h EMA50 AND volume spike
            elif price < s3_val and price < ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to pivot point (PP)
            if position == 1 and price < pp_val:
                exit_signal = True
            elif position == -1 and price > pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 09:16
