# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.500 | +38.4% | -4.4% | 230 | KEEP |
| ETHUSDT | 0.770 | +54.0% | -5.0% | 212 | KEEP |
| SOLUSDT | 0.519 | +54.5% | -11.8% | 172 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.511 | -4.2% | -8.0% | 80 | DISCARD |
| ETHUSDT | 0.912 | +16.8% | -4.9% | 79 | KEEP |
| SOLUSDT | -0.384 | +1.9% | -7.0% | 58 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot breakout with 12h trend filter and volume spike.
# Long when price breaks above R3 with 12h EMA50 uptrend and volume > 2x average.
# Short when price breaks below S3 with 12h EMA50 downtrend and volume > 2x average.
# Exit when price crosses the Camarilla Pivot (PP).
# Uses tight entry conditions to limit trades and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # PP = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    typical_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    pp_12h = typical_12h
    r3_12h = close_12h + range_12h * 1.1 / 2
    s3_12h = close_12h - range_12h * 1.1 / 2
    
    # Align 12h indicators to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Get volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA50, pivot levels, and volume MA20
    start_idx = max(ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(pp_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: break above R3 with 12h EMA50 uptrend and volume spike
            if (price > r3_12h_aligned[i] and 
                price > ema_12h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below S3 with 12h EMA50 downtrend and volume spike
            elif (price < s3_12h_aligned[i] and 
                  price < ema_12h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Pivot Point
            if price < pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Pivot Point
            if price > pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-05-04 12:02
