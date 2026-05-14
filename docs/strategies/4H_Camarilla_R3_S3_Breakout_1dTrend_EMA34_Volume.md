# Strategy: 4H_Camarilla_R3_S3_Breakout_1dTrend_EMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.503 | +45.0% | -8.9% | 184 | KEEP |
| ETHUSDT | 0.024 | +20.0% | -12.9% | 184 | KEEP |
| SOLUSDT | 0.840 | +115.7% | -16.0% | 158 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.054 | -4.5% | -6.9% | 74 | DISCARD |
| ETHUSDT | 0.824 | +19.7% | -12.5% | 62 | KEEP |
| SOLUSDT | 0.066 | +6.4% | -8.3% | 53 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
4H_Camarilla_R3_S3_Breakout_1dTrend_EMA34_Volume
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
This strategy targets breakouts from key daily pivot levels (R3/S3) in the direction
of the daily trend, filtered by volume spikes to avoid false breakouts. Works in both
bull and bear markets by only taking breakouts aligned with the higher timeframe trend.
Designed for low frequency (20-50 trades/year) to minimize fee drag.
"""

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and EMA trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Calculate Camarilla levels from previous 1d bar ---
    # Camarilla uses previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day values (shifted by 1)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    # First day has no previous, set to same day values
    phigh[0] = high_1d[0]
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    # Camarilla calculations
    R3 = pclose + (phigh - plow) * 1.1 / 4
    S3 = pclose - (phigh - plow) * 1.1 / 4
    
    # Align Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: Breakout of R3/S3 with volume and trend alignment
        long_entry = (close[i] > R3_aligned[i]) and vol_spike[i] and (close[i] > ema_34_1d_aligned[i])
        short_entry = (close[i] < S3_aligned[i]) and vol_spike[i] and (close[i] < ema_34_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: Opposite breakout or trend reversal
            if position == 1:
                # Exit if price breaks below S3 or trend turns down
                if (close[i] < S3_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price breaks above R3 or trend turns up
                if (close[i] > R3_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 05:27
