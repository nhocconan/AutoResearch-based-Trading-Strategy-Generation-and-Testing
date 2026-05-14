# Strategy: 6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_PPExit_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.540 | +43.2% | -6.7% | 129 | PASS |
| ETHUSDT | 0.108 | +25.0% | -11.7% | 111 | PASS |
| SOLUSDT | 0.845 | +105.7% | -18.1% | 106 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.522 | +1.6% | -4.7% | 48 | FAIL |
| ETHUSDT | 0.787 | +17.3% | -7.4% | 43 | PASS |
| SOLUSDT | -0.259 | +1.9% | -8.1% | 42 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike.
Long when price breaks above Camarilla R3 AND price > 1d EMA50 AND volume > 2.0x 24-period average.
Short when price breaks below Camarilla S3 AND price < 1d EMA50 AND volume > 2.0x 24-period average.
Exit when price reverts to Camarilla pivot (PP) OR ATR trailing stop (2.0*ATR from extreme).
Uses 1d HTF for trend alignment and Camarilla levels from daily.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day (using daily data)
    # Use previous day's data to avoid look-ahead
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)
    # First value will be NaN due to roll, handled by min_periods in align
    
    # Camarilla formula: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    rng = high_1d_prev - low_1d_prev
    # R3 = PP + 1.1 * Range / 2, S3 = PP - 1.1 * Range / 2
    r3 = pp + 1.1 * rng / 2.0
    s3 = pp - 1.1 * rng / 2.0
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 6h volume average (24-period = 4 days) for spike filter
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(14) for 6h trailing stop calculation
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
    start_idx = max(24, 50, 1)  # vol_ma24, ema_50_1d, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pp_val = pp_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 1d EMA50 AND volume spike
            if price > r3_val and price > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below S3 AND price < 1d EMA50 AND volume spike
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

name = "6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_PPExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 09:28
