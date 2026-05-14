# Strategy: 12h_Camarilla_R1S1_1dEMA34_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.039 | +18.6% | -10.6% | 137 | FAIL |
| ETHUSDT | 0.082 | +23.6% | -13.0% | 113 | PASS |
| SOLUSDT | 0.419 | +56.1% | -31.0% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.635 | +15.6% | -6.4% | 41 | PASS |
| SOLUSDT | -0.259 | +1.2% | -14.5% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses 12h Camarilla pivot levels (R1/S1) for tight structure - high-probability breakouts in both bull/bear
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.5x average volume)
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets (continuation at R2/S2) and bear markets (continuation at R1/S1)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R1S1_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot points (based on prior completed 1d bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Pivot + Range * 1.1/12
    # S1 = Pivot - Range * 1.1/12
    # R2 = Pivot + Range * 1.1/6
    # S2 = Pivot - Range * 1.1/6
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r1_1d = pivot_1d + (range_1d * 1.1 / 12.0)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12.0)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6.0)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6.0)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    r1_1d_shifted = np.roll(r1_1d, 1)
    s1_1d_shifted = np.roll(s1_1d, 1)
    r2_1d_shifted = np.roll(r2_1d, 1)
    s2_1d_shifted = np.roll(s2_1d, 1)
    r1_1d_shifted[0] = np.nan
    s1_1d_shifted[0] = np.nan
    r2_1d_shifted[0] = np.nan
    s2_1d_shifted[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d_shifted)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d_shifted)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d_shifted)
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND price > 1d EMA34 AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 AND price < 1d EMA34 AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S1 OR price crosses below 1d EMA34
            if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R1 OR price crosses above 1d EMA34
            if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 09:59
