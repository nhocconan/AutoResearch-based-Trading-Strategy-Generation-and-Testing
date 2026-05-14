# Strategy: 4h_1dCamarilla_R3S3_EMA200_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.020 | +21.7% | -7.0% | 389 | PASS |
| ETHUSDT | 0.130 | +25.7% | -7.8% | 363 | PASS |
| SOLUSDT | 0.309 | +39.3% | -17.9% | 348 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.516 | -3.7% | -6.7% | 131 | FAIL |
| ETHUSDT | 0.245 | +8.5% | -5.9% | 128 | PASS |
| SOLUSDT | -0.323 | +2.7% | -5.9% | 112 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels with trend filter and volume confirmation
# Long when price breaks above R3 with price > 200-bar EMA and volume > 1.5x average
# Short when price breaks below S3 with price < 200-bar EMA and volume > 1.5x average
# Uses 1d pivot levels for institutional support/resistance, EMA200 for trend filter, volume for confirmation
# Target: 25-40 trades per year (100-160 over 4 years) with 0.25 position sizing
# Works in bull markets via breakouts above resistance and in bear markets via breakdowns below support

name = "4h_1dCamarilla_R3S3_EMA200_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 on 4h close (needs 200 bars)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema200[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with uptrend and volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema200[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with downtrend and volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema200[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (support break)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (resistance break)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 22:20
