# Strategy: 4H_Camarilla_R3_S3_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.494 | +36.5% | -7.0% | 283 | PASS |
| ETHUSDT | 0.230 | +29.5% | -7.6% | 268 | PASS |
| SOLUSDT | 0.315 | +38.0% | -10.3% | 232 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.526 | -3.9% | -5.0% | 119 | FAIL |
| ETHUSDT | 1.843 | +28.6% | -4.6% | 94 | PASS |
| SOLUSDT | 1.815 | +26.2% | -3.9% | 79 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above 4h Camarilla R3 level AND price > 1d EMA34 (uptrend) AND volume > 2.0x average.
Short when price breaks below 4h Camarilla S3 level AND price < 1d EMA34 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 4h Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA34).
Uses 4h timeframe with tighter entry conditions (Camarilla R3/S3 levels) to limit trades to 75-150 over 4 years.
1d EMA34 provides smoother trend filter than 12h EMA50. Volume spike ensures high-conviction breakouts.
Target: 90-120 trades over 4 years (22-30/year) to stay within proven working range and avoid fee drag.
Works in both bull and bear markets: trend filter prevents counter-trend trades, volume confirms conviction.
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
    
    # Calculate 4h Camarilla levels (R3, S3, PP) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels on 4h (based on previous 4h bar's OHLC)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    # Set first value to NaN (no previous bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_pp = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    camarilla_r3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4.0
    camarilla_s3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4.0
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r3_val and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s3_val and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Camarilla PP OR price breaks below 1d EMA34 (trend reversal)
                if price <= pp_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Camarilla PP OR price breaks above 1d EMA34 (trend reversal)
                if price >= pp_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:56
