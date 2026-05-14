# Strategy: 1h_Camarilla_R3_S3_Breakout_4hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.316 | +5.5% | -12.1% | 464 | FAIL |
| ETHUSDT | 0.128 | +26.2% | -14.6% | 444 | PASS |
| SOLUSDT | 0.104 | +22.5% | -31.1% | 386 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.067 | +6.4% | -10.5% | 143 | PASS |
| SOLUSDT | 0.188 | +8.3% | -8.3% | 131 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and 1d for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d Camarilla levels from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe (using previous 1d bar's values)
    r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 2.0x 24-period average (tightened for 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 4h EMA20 (uptrend) AND volume surge
            if close[i] > r3_1h[i] and close[i] > ema_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND below 4h EMA20 (downtrend) AND volume surge
            elif close[i] < s3_1h[i] and close[i] < ema_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below 4h EMA20 (trend change)
            if close[i] < s3_1h[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above 4h EMA20 (trend change)
            if close[i] > r3_1h[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals
```

## Last Updated
2026-05-11 14:14
