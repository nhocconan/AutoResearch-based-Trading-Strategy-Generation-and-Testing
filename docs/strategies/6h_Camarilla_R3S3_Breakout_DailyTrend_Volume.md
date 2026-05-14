# Strategy: 6h_Camarilla_R3S3_Breakout_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.396 | +40.1% | -10.8% | 115 | PASS |
| ETHUSDT | 0.214 | +31.4% | -12.8% | 111 | PASS |
| SOLUSDT | 0.660 | +89.2% | -18.9% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.989 | -4.5% | -8.3% | 47 | FAIL |
| ETHUSDT | 1.291 | +29.0% | -6.4% | 33 | PASS |
| SOLUSDT | -0.277 | +0.8% | -14.2% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot breakout with daily trend filter and volume confirmation
# We go long when price breaks above R3 with daily EMA(34) uptrend and volume spike.
# We go short when price breaks below S3 with daily EMA(34) downtrend and volume spike.
# Uses 6h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Camarilla pivots provide mathematically derived support/resistance levels.
# Daily trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.

name = "6h_Camarilla_R3S3_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from daily data
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_vals = df_1d['close'].values
    
    camarilla_r3 = daily_close_vals + 1.1 * (daily_high - daily_low) / 2
    camarilla_s3 = daily_close_vals - 1.1 * (daily_high - daily_low) / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + daily uptrend + volume spike
            if (not np.isnan(r3_level) and close[i] > r3_level and 
                close[i] > ema34_1d_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + daily downtrend + volume spike
            elif (not np.isnan(s3_level) and close[i] < s3_level and 
                  close[i] < ema34_1d_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR daily trend turns down
            if (not np.isnan(s3_level) and close[i] < s3_level) or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR daily trend turns up
            if (not np.isnan(r3_level) and close[i] > r3_level) or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 12:50
