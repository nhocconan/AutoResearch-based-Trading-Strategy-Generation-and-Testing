# Strategy: 4h_Camarilla_R1S1_12hEMA50_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.307 | +6.8% | -10.0% | 360 | FAIL |
| ETHUSDT | 0.100 | +24.5% | -9.9% | 331 | PASS |
| SOLUSDT | 0.781 | +104.3% | -28.0% | 307 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.004 | +22.6% | -6.8% | 109 | PASS |
| SOLUSDT | 0.892 | +21.0% | -10.2% | 103 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation (>1.6x 20 EMA volume)
# Uses 12h Camarilla pivot levels (R1/S1) for tight structure - high-probability breakouts in both bull/bear
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.6x average volume)
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 75-150 total trades over 4 years = 19-38/year for 4h timeframe
# Works in bull markets (continuation at R2/S2) and bear markets (continuation at R1/S1)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "4h_Camarilla_R1S1_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot points (based on prior completed 12h bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Pivot + Range * 1.1/12
    # S1 = Pivot - Range * 1.1/12
    # R2 = Pivot + Range * 1.1/6
    # S2 = Pivot - Range * 1.1/6
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    r1_12h = pivot_12h + (range_12h * 1.1 / 12.0)
    s1_12h = pivot_12h - (range_12h * 1.1 / 12.0)
    r2_12h = pivot_12h + (range_12h * 1.1 / 6.0)
    s2_12h = pivot_12h - (range_12h * 1.1 / 6.0)
    
    # Shift by 1 to use only prior completed 12h bar (no look-ahead)
    r1_12h_shifted = np.roll(r1_12h, 1)
    s1_12h_shifted = np.roll(s1_12h, 1)
    r2_12h_shifted = np.roll(r2_12h, 1)
    s2_12h_shifted = np.roll(s2_12h, 1)
    r1_12h_shifted[0] = np.nan
    s1_12h_shifted[0] = np.nan
    r2_12h_shifted[0] = np.nan
    s2_12h_shifted[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h_shifted)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h_shifted)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h_shifted)
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND price > 12h EMA50 AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 AND price < 12h EMA50 AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S1 OR price crosses below 12h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R1 OR price crosses above 12h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 09:58
