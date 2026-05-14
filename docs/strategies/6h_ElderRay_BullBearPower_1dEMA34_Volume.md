# Strategy: 6h_ElderRay_BullBearPower_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.052 | +16.8% | -14.8% | 145 | FAIL |
| ETHUSDT | 0.124 | +25.9% | -16.5% | 151 | PASS |
| SOLUSDT | 0.967 | +163.0% | -28.0% | 139 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.070 | +6.3% | -8.6% | 56 | PASS |
| SOLUSDT | -0.052 | +4.0% | -14.7% | 47 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long: Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume > 1.5x average
# Short: Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume > 1.5x average
# Uses 13-period EMA for Elder Ray calculation (standard setting)
# 1d EMA34 filter ensures trading with higher timeframe trend (avoids counter-trend trades)
# Volume confirmation filters out low-conviction moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_ElderRay_BullBearPower_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate 1d EMA34 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.5x 24-period average (1 day at 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative)
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive)
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR price < 1d EMA34
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR price > 1d EMA34
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 05:10
