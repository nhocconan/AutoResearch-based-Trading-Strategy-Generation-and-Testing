# Strategy: 12h_Camarilla_R3S3_1dEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.033 | +20.3% | -4.3% | 69 | FAIL |
| ETHUSDT | 0.026 | +21.7% | -6.6% | 61 | PASS |
| SOLUSDT | 0.059 | +21.6% | -18.7% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.750 | +13.9% | -3.6% | 23 | PASS |
| SOLUSDT | -0.345 | +2.1% | -10.1% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 12h Camarilla R3 level in 1d uptrend (price > EMA50) with volume spike.
# Short when price breaks below 12h Camarilla S3 level in 1d downtrend (price < EMA50) with volume spike.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years.
# Camarilla levels provide clear support/resistance structure, 1d EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading with
# the 1d trend, avoiding counter-trend whipsaws. Designed for 12h timeframe to minimize fee drag.

name = "12h_Camarilla_R3S3_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous day's high/low/close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3_12h = close_12h + (high_12h - low_12h) * 1.1 / 4
    camarilla_s3_12h = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 15m timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close_val > camarilla_r3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close_val < camarilla_s3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (reversal signal)
            if close_val < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (reversal signal)
            if close_val > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 09:29
