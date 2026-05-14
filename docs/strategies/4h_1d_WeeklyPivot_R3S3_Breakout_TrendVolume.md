# Strategy: 4h_1d_WeeklyPivot_R3S3_Breakout_TrendVolume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.023 | +19.8% | -14.2% | 82 | PASS |
| ETHUSDT | 0.248 | +35.7% | -14.2% | 76 | PASS |
| SOLUSDT | 1.085 | +212.2% | -21.2% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.402 | -3.3% | -6.0% | 24 | FAIL |
| ETHUSDT | 0.524 | +13.0% | -6.5% | 15 | PASS |
| SOLUSDT | -0.018 | +5.3% | -6.5% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_WeeklyPivot_R3S3_Breakout_TrendVolume
Hypothesis: Price breaking above weekly R3 or below weekly S3 with daily trend confirmation (EMA34) and volume spike. Uses weekly pivot levels as strong support/resistance. In uptrend (price > EMA34 daily), buy breakouts above R3; in downtrend (price < EMA34 daily), sell breakdowns below S3. Volume confirms institutional interest. Designed for 4h timeframe with weekly pivot structure and daily trend filter to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets by capturing strong momentum moves after breaking key weekly levels.
"""

name = "4h_1d_WeeklyPivot_R3S3_Breakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using previous week's OHLC
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Shift to use previous week's data (avoid look-ahead)
    w_high_prev = np.roll(w_high, 1)
    w_low_prev = np.roll(w_low, 1)
    w_close_prev = np.roll(w_close, 1)
    # First period: use current values to avoid NaN
    w_high_prev[0] = w_high[0]
    w_low_prev[0] = w_low[0]
    w_close_prev[0] = w_close[0]
    
    # Calculate weekly pivot point
    w_pivot = (w_high_prev + w_low_prev + w_close_prev) / 3.0
    # Calculate weekly R3 and S3 levels (more extreme levels)
    w_r3 = w_close_prev + (1.1/4) * (w_high_prev - w_low_prev)
    w_s3 = w_close_prev - (1.1/4) * (w_high_prev - w_low_prev)
    
    # Align weekly R3/S3 to 4h timeframe
    w_r3_aligned = align_htf_to_ltf(prices, df_1w, w_r3)
    w_s3_aligned = align_htf_to_ltf(prices, df_1w, w_s3)
    w_pivot_aligned = align_htf_to_ltf(prices, df_1w, w_pivot)
    
    # Daily trend filter (EMA 34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(w_r3_aligned[i]) or np.isnan(w_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above weekly R3 + above daily EMA34 + volume spike
            if (close[i] > w_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3 + below daily EMA34 + volume spike
            elif (close[i] < w_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to weekly pivot or trend reversal
            if position == 1:
                # Exit long: price returns to weekly pivot OR trend turns down
                if (close[i] <= w_pivot_aligned[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly pivot OR trend turns up
                if (close[i] >= w_pivot_aligned[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 08:03
