# Strategy: 4h_Camarilla_R3_S3_Breakout_Trend_Volume_12h

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.102 | +13.1% | -16.3% | 215 | FAIL |
| ETHUSDT | 0.062 | +21.4% | -16.0% | 197 | PASS |
| SOLUSDT | 0.667 | +101.4% | -23.8% | 183 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.150 | +28.4% | -7.2% | 62 | PASS |
| SOLUSDT | 0.038 | +5.5% | -14.1% | 72 | PASS |

## Code
```python
# 4h_Camarilla_R3_S3_Breakout_Trend_Volume_12h
# Hypothesis: Daily Camarilla R3/S3 levels act as strong support/resistance on 4h chart.
# Breakouts above R3 or below S3 with volume confirmation and 12h EMA trend filter capture momentum.
# Uses 4h for execution and 12h EMA for trend direction. Target ~30-50 trades/year to avoid fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns against trend filtered by 12h EMA).

name = "4h_Camarilla_R3_S3_Breakout_Trend_Volume_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 2.0
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 2.0
    
    # Align daily Camarilla levels to 4h chart (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above daily R3 with volume confirmation and 12h EMA uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_filter[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below daily S3 with volume confirmation and 12h EMA downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_filter[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily S3 or breaks below 12h EMA
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily R3 or breaks above 12h EMA
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 09:29
