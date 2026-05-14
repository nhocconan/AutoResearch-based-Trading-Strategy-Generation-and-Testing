# Strategy: 6h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.165 | +28.6% | -14.7% | 76 | PASS |
| ETHUSDT | 0.193 | +31.2% | -16.8% | 75 | PASS |
| SOLUSDT | 0.837 | +146.9% | -25.7% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.891 | -5.5% | -11.3% | 31 | FAIL |
| ETHUSDT | 0.653 | +19.4% | -8.4% | 25 | PASS |
| SOLUSDT | -0.263 | -1.3% | -22.0% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 12h EMA50 (trend reversal)
# Uses 6h primary timeframe with 12h HTF for trend filter and Camarilla structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) based on proven Camarilla breakout performance
# Works in both bull and bear markets by following the 12h trend while using 6h for entry timing

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels on 12h data (based on previous day's OHLC)
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = C + 1.1*(H-L)/2 * 1.1/2 = C + 1.1*(H-L)/2 * 0.55
    # S3 = C - 1.1*(H-L)/2 * 1.1/2 = C - 1.1*(H-L)/2 * 0.55
    # Simplified: R3 = close + 1.1 * (high - low) * 0.55, S3 = close - 1.1 * (high - low) * 0.55
    camarilla_r3 = df_12h['close'].values + 1.1 * (df_12h['high'].values - df_12h['low'].values) * 0.55
    camarilla_s3 = df_12h['close'].values - 1.1 * (df_12h['high'].values - df_12h['low'].values) * 0.55
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 05:57
