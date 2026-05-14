# Strategy: 6h_ElderRay_1dEMA34_Trend_VolumeConfirmation_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.249 | +6.8% | -18.9% | 36 | FAIL |
| ETHUSDT | 0.003 | +18.3% | -18.5% | 28 | PASS |
| SOLUSDT | 0.607 | +90.2% | -22.6% | 33 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.135 | +7.4% | -9.1% | 12 | PASS |
| SOLUSDT | -0.606 | -8.6% | -18.5% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 1d).
- Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling.
- Trend filter: Only trade in direction of 1d EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current 6h volume > 1.5x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strength (Bull Power > 0) in uptrend, in bear via selling weakness (Bear Power < 0) in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray) and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h (each 1d bar = 4x 6h bars)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA34 trend
            if i > 0 and not np.isnan(ema_34_1d_aligned[i-1]):
                ema34_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Long when Bull Power > 0 and rising with volume spike
                    if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Short when Bear Power < 0 and falling with volume spike
                    if bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or opposite signal
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or opposite signal
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 08:49
