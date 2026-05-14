# Strategy: 4H_Camarilla_R4_S4_Breakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.177 | +26.4% | -4.8% | 247 | PASS |
| ETHUSDT | 0.244 | +29.8% | -6.7% | 228 | PASS |
| SOLUSDT | 0.165 | +28.3% | -11.0% | 187 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.189 | -1.2% | -4.7% | 88 | FAIL |
| ETHUSDT | 1.447 | +22.2% | -3.4% | 79 | PASS |
| SOLUSDT | 0.925 | +15.1% | -4.1% | 63 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above 4h Camarilla R4 level AND price > 12h EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below 4h Camarilla S4 level AND price < 12h EMA50 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 4h Camarilla pivot point (PP) or trend reverses (price crosses 12h EMA50).
Uses 4h timeframe with tighter entry conditions (Camarilla R4/S4 are extreme resistance/support) to limit trades.
12h EMA50 provides smoother trend filter than 1d EMA34. Volume spike ensures high-conviction breakouts.
Target: 60-120 trades over 4 years (15-30/year) to stay within proven working range and avoid fee drag.
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
    
    # Calculate 4h Camarilla levels (R4, S4, PP) - ONCE before loop
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
    camarilla_r4 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 2.0 * 2.0  # R4 = R3 * 2
    camarilla_s4 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 2.0 * 2.0  # S4 = S3 * 2
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R4 AND price > 12h EMA50 (uptrend) AND volume spike
            if (price > r4_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Camarilla S4 AND price < 12h EMA50 (downtrend) AND volume spike
            elif (price < s4_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Camarilla PP OR price breaks below 12h EMA50 (trend reversal)
                if price <= pp_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Camarilla PP OR price breaks above 12h EMA50 (trend reversal)
                if price >= pp_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R4_S4_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 01:54
