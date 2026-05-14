# Strategy: 4h_12hCamarilla_R3S3_Breakout_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.110 | +25.0% | -8.9% | 200 | KEEP |
| ETHUSDT | 0.470 | +48.5% | -15.8% | 181 | KEEP |
| SOLUSDT | 0.613 | +76.9% | -25.0% | 148 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.636 | -0.2% | -5.9% | 68 | DISCARD |
| ETHUSDT | 0.642 | +16.0% | -8.5% | 57 | KEEP |
| SOLUSDT | -0.278 | +1.2% | -15.3% | 55 | DISCARD |

## Code
```python
#!/usr/bin/env python3
# 4h_12hCamarilla_R3S3_Breakout_12hEMA50_Trend_Volume
# Uses 12h Camarilla pivot levels (R3/S3) as breakout levels with 12h trend filter (EMA50)
# and 4h volume confirmation. Designed for 4h timeframe to capture major pivot breaks
# with trend alignment, working in both bull and bear markets by following the 12h trend.
# Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing.

name = "4h_12hCamarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3 = pp + range_12h * 1.1 / 2
    s3 = pp - range_12h * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_12h, r3)
    s3_4h = align_htf_to_ltf(prices, df_12h, s3)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume
            if close[i] > r3_4h[i] and close[i] > ema_50_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume
            elif close[i] < s3_4h[i] and close[i] < ema_50_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to EMA50 or breaks below S3
            if close[i] < ema_50_4h[i] or close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to EMA50 or breaks above R3
            if close[i] > ema_50_4h[i] or close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
```

## Last Updated
2026-05-07 00:02
