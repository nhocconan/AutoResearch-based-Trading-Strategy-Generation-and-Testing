# Strategy: 12h_1D_Camarilla_R1_S1_Breakout_Trend_With_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.588 | +6.4% | -8.1% | 127 | FAIL |
| ETHUSDT | 0.120 | +25.2% | -7.0% | 102 | PASS |
| SOLUSDT | 0.357 | +44.0% | -15.9% | 101 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.589 | +12.8% | -5.7% | 42 | PASS |
| SOLUSDT | -0.477 | +0.3% | -6.6% | 40 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_1D_Camarilla_R1_S1_Breakout_Trend_With_Volume_Filter
# Hypothesis: Breakouts above daily Camarilla R1 in uptrend (price > EMA34) and breakdowns below S1 in downtrend (price < EMA34), with volume confirmation (volume > 1.8x 20-period average). Uses 12h timeframe to reduce trade frequency, targeting 12-37 trades per year. Designed to work in both bull and bear markets by requiring trend alignment and avoiding whipsaw through volume confirmation.

name = "12h_1D_Camarilla_R1_S1_Breakout_Trend_With_Volume_Filter"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    camarilla_r1[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    camarilla_s1[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(50, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Price breaks above R1 with volume confirmation in uptrend (price > EMA34)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               high[i] > camarilla_r1_aligned[i] and volume_confirmed[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation in downtrend (price < EMA34)
            elif camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
                 low[i] < camarilla_s1_aligned[i] and volume_confirmed[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 or trend weakens (price < EMA34)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               low[i] < camarilla_r1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 3  # 3-bar cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 or trend weakens (price > EMA34)
            if camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
               high[i] > camarilla_s1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 3  # 3-bar cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 09:37
