# Strategy: 4H_Camarilla_R3_S3_1DTrend_VolumeSpike_MomentumExit

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.264 | +28.7% | -5.5% | 261 | PASS |
| ETHUSDT | 0.041 | +22.3% | -10.3% | 250 | PASS |
| SOLUSDT | 0.445 | +45.8% | -13.8% | 222 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.173 | -7.0% | -8.3% | 104 | FAIL |
| ETHUSDT | 0.140 | +7.2% | -8.8% | 89 | PASS |
| SOLUSDT | 0.698 | +12.1% | -4.2% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_1DTrend_VolumeSpike_MomentumExit
# Hypothesis: Combines Camarilla R3/S3 breakout with 1-day EMA trend, volume spike, and momentum-based exit.
# Uses MOMENTUM (10-period ROC) to exit early when momentum fades, reducing whipsaw in sideways markets.
# Designed for 4h timeframe with low trade frequency (<50/year) and strong performance in both bull and bear regimes.
# Target: 20-50 trades per year per symbol with clear entry/exit rules.

name = "4H_Camarilla_R3_S3_1DTrend_VolumeSpike_MomentumExit"
timeframe = "4h"
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
    
    # Get 1d data for Camarillo pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Momentum filter: 10-period ROC for exit signal
    roc_period = 10
    roc = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(roc_period, n):
        if close[i - roc_period] != 0:
            roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, roc_period)  # Ensure we have volume MA and ROC data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + Uptrend (price > EMA34) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + Downtrend (price < EMA34) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Momentum reversal: ROC crosses zero against position
            # 2. Price returns inside pivot range (reversion to mean)
            momentum_exit = (position == 1 and roc[i] < 0) or (position == -1 and roc[i] > 0)
            price_inside = (close[i] < r3_aligned[i] and close[i] > s3_aligned[i])
            
            if momentum_exit or price_inside:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 03:26
