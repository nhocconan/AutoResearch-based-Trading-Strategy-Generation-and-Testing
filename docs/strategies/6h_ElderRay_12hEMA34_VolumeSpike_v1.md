# Strategy: 6h_ElderRay_12hEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.018 | +17.9% | -12.2% | 181 | FAIL |
| ETHUSDT | 0.028 | +19.2% | -17.4% | 174 | PASS |
| SOLUSDT | 1.048 | +191.9% | -28.7% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.030 | +5.4% | -11.6% | 57 | PASS |
| SOLUSDT | -0.799 | -10.2% | -21.9% | 59 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h, HTF: 12h for EMA34 trend alignment.
- Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close) on 6h.
- Trend filter: only long when 6h close > 12h EMA34, only short when 6h close < 12h EMA34.
- Volume confirmation: current 6h volume > 1.8 * 30-period 6h volume MA.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Exit: Elder Ray power reverses sign (Bull Power < 0 for long exit, Bear Power > 0 for short exit).
- Works in bull via trend alignment, in bear via power reversal signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 13-period EMA for Elder Ray (on 6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.8 * 30-period volume MA
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 13)  # Need 12h EMA34, volume MA, EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND uptrend AND volume spike
            if bull_power[i] > 0 and close[i] > ema_34_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND downtrend AND volume spike
            elif bear_power[i] < 0 and close[i] < ema_34_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power becomes negative (momentum fading)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power becomes positive (momentum fading)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 07:08
