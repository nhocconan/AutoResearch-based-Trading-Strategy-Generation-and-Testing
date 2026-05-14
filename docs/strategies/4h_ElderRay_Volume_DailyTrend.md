# Strategy: 4h_ElderRay_Volume_DailyTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.131 | +26.2% | -11.9% | 240 | PASS |
| ETHUSDT | 0.174 | +29.3% | -15.9% | 230 | PASS |
| SOLUSDT | 0.938 | +148.4% | -19.2% | 189 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.177 | -6.5% | -7.8% | 95 | FAIL |
| ETHUSDT | 1.184 | +28.4% | -12.8% | 81 | PASS |
| SOLUSDT | -0.544 | -4.2% | -18.2% | 70 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h Elder Ray + Volume Spike + Daily Trend Filter
# Hypothesis: Elder Ray (bull/bear power) detects institutional strength. Bull Power > 0 indicates buying pressure,
# Bear Power < 0 indicates selling pressure. Combined with volume spikes and daily EMA trend filter,
# this captures strong momentum moves while avoiding chop. Designed for low trade frequency (~20-30/year).
name = "4h_ElderRay_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Elder Ray on 4h: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h[i]) or np.isnan(ema_13[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (buying pressure) + volume spike + price above daily EMA34 (uptrend)
            if (bull_power[i] > 0 and 
                vol_spike[i] and
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (selling pressure) + volume spike + price below daily EMA34 (downtrend)
            elif (bear_power[i] < 0 and 
                  vol_spike[i] and
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power turns negative (loss of buying pressure)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive (loss of selling pressure)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 06:44
