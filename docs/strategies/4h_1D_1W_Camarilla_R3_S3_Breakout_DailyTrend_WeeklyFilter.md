# Strategy: 4h_1D_1W_Camarilla_R3_S3_Breakout_DailyTrend_WeeklyFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.524 | +33.8% | -4.2% | 113 | PASS |
| ETHUSDT | 0.100 | +24.0% | -7.2% | 90 | PASS |
| SOLUSDT | 0.228 | +30.1% | -11.0% | 70 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.426 | -2.1% | -3.4% | 34 | FAIL |
| ETHUSDT | 0.987 | +12.7% | -2.6% | 28 | PASS |
| SOLUSDT | 0.203 | +7.4% | -4.5% | 16 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1D_1W_Camarilla_R3_S3_Breakout_DailyTrend_WeeklyFilter
# Hypothesis: 4-hour breakouts from daily Camarilla R3/S3 levels with daily trend filter and weekly trend confirmation.
# Only takes long when both daily and weekly trends are up, short when both are down.
# Uses volume spike to confirm institutional participation.
# Targets 20-50 trades per year by requiring confluence of daily trend, weekly trend, daily level break, and volume spike.

name = "4h_1D_1W_Camarilla_R3_S3_Breakout_DailyTrend_WeeklyFilter"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Camarilla R3 and S3 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R3_1d = prev_close_1d + 1.1 * rang_1d * 3.0 / 4
    S3_1d = prev_close_1d - 1.1 * rang_1d * 3.0 / 4
    
    # Align daily levels to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above daily EMA34 (daily uptrend) + price above weekly EMA34 (weekly uptrend)
            if (close[i] > R3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below daily EMA34 (daily downtrend) + price below weekly EMA34 (weekly downtrend)
            elif (close[i] < S3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below daily EMA34 OR weekly EMA34
            if (close[i] < R3_1d_aligned[i] and close[i] > S3_1d_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i] or \
               close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above daily EMA34 OR weekly EMA34
            if (close[i] < R3_1d_aligned[i] and close[i] > S3_1d_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i] or \
               close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 11:02
