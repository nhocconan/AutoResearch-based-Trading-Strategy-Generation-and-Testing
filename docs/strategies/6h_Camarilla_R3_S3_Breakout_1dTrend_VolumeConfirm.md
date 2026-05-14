# Strategy: 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.189 | +13.9% | -8.4% | 351 | FAIL |
| ETHUSDT | 0.101 | +24.6% | -10.2% | 334 | PASS |
| SOLUSDT | 0.884 | +105.4% | -13.8% | 303 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.563 | +13.6% | -7.4% | 116 | PASS |
| SOLUSDT | 0.287 | +9.5% | -10.0% | 112 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakout with volume spike, filtered by 1d EMA34 trend.
Long when price breaks above R3 with volume > 1.5x average and close > 1d EMA34.
Short when price breaks below S3 with volume > 1.5x average and close < 1d EMA34.
Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 trades over 4 years.
Works in bull/bear via 1d trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 6h
    # Based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid nan on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    r4 = prev_close + range_hl * 1.1 / 2
    s4 = prev_close - range_hl * 1.1 / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 with volume confirmation and 1d uptrend
        long_condition = (close[i] > r3[i]) and volume_confirm[i] and (close[i] > ema_34_1d_aligned[i])
        # Short logic: break below S3 with volume confirmation and 1d downtrend
        short_condition = (close[i] < s3[i]) and volume_confirm[i] and (close[i] < ema_34_1d_aligned[i])
        
        # Exit logic: opposite Camarilla level touch or trend reversal
        exit_long = (close[i] < s3[i]) or (close[i] < ema_34_1d_aligned[i])
        exit_short = (close[i] > r3[i]) or (close[i] > ema_34_1d_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-26 17:09
