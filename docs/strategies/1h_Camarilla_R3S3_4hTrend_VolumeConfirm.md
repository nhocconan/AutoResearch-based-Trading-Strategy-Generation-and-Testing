# Strategy: 1h_Camarilla_R3S3_4hTrend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.345 | +3.6% | -14.7% | 330 | DISCARD |
| ETHUSDT | 0.056 | +21.5% | -14.5% | 317 | KEEP |
| SOLUSDT | 0.255 | +38.3% | -29.1% | 306 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.040 | +6.0% | -10.2% | 98 | KEEP |
| SOLUSDT | -0.543 | -3.8% | -13.2% | 98 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND 4h close > 4h EMA50 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below S3 AND 4h close < 4h EMA50 (downtrend) AND volume > 1.5x 20 EMA
# Uses 1h for precise entry timing, 4h for trend direction to avoid counter-trend trades.
# Discrete sizing (0.20) to minimize fee churn. Target: 15-37 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1h_Camarilla_R3S3_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 4h uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND 4h downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR 4h trend changes to downtrend
            if (close[i] < s3_aligned[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 OR 4h trend changes to uptrend
            if (close[i] > r3_aligned[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-05-05 00:02
