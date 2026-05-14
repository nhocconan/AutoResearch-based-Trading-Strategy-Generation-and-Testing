# Strategy: 12h_Camarilla_R3_S3_DailyTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.015 | +20.5% | -6.2% | 78 | FAIL |
| ETHUSDT | 0.126 | +25.8% | -13.0% | 66 | PASS |
| SOLUSDT | 0.221 | +33.3% | -19.9% | 63 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.111 | +7.1% | -5.3% | 28 | PASS |
| SOLUSDT | -0.693 | -2.7% | -12.5% | 25 | FAIL |

## Code
```python
# 12h_Camarilla_R3_S3_DailyTrend_VolumeSpike
# Hypothesis: Price tends to rebound from strong intraday support/resistance levels (Camarilla R3/S3)
# with confirmation from higher timeframe trend and volume spikes. Works in both bull and bear markets
# by capturing mean reversion from overextended moves. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from PRIOR day's OHLC (avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formula: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    Pivot_12h = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(Pivot_12h[i]) or np.isnan(ema34_12h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R3 with volume spike and daily uptrend
            if (close[i] > R3_12h[i] and volume_spike[i] and close[i] > ema34_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S3 with volume spike and daily downtrend
            elif (close[i] < S3_12h[i] and volume_spike[i] and close[i] < ema34_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to pivot or trend fails
            if (close[i] <= Pivot_12h[i] or close[i] < ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot or trend fails
            if (close[i] >= Pivot_12h[i] or close[i] > ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-27 17:29
