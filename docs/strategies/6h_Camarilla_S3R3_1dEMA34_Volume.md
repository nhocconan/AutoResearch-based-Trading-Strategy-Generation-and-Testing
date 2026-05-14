# Strategy: 6h_Camarilla_S3R3_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.402 | -0.5% | -16.8% | 71 | FAIL |
| ETHUSDT | 0.208 | +32.2% | -16.3% | 67 | PASS |
| SOLUSDT | 0.362 | +51.8% | -25.1% | 69 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.035 | +28.8% | -8.6% | 20 | PASS |
| SOLUSDT | 0.412 | +13.9% | -8.7% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d pivot levels (Camarilla S3/R3) with volume spike and 1d EMA34 trend filter
# - Long when price touches or exceeds S3 level with volume spike and price above 1d EMA34
# - Short when price touches or exceeds R3 level with volume spike and price below 1d EMA34
# - Exit when price crosses back below/above 1d EMA34
# - Designed to capture mean-reversion bounces at strong intraday support/resistance levels
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_Camarilla_S3R3_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (S1, S2, S3, R1, R2, R3)
    # Formula: P = (H + L + C) / 3, Range = H - L
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    s1 = close_1d - (range_hl * 1.0 / 6)
    s2 = close_1d - (range_hl * 2.0 / 6)
    s3 = close_1d - (range_hl * 3.0 / 6)
    r1 = close_1d + (range_hl * 1.0 / 6)
    r2 = close_1d + (range_hl * 2.0 / 6)
    r3 = close_1d + (range_hl * 3.0 / 6)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(s3_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or exceeds S3 with volume spike and above EMA34
            if low[i] <= s3_6h[i] and volume_spike[i] and close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or exceeds R3 with volume spike and below EMA34
            elif high[i] >= r3_6h[i] and volume_spike[i] and close[i] < ema_34_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 22:59
