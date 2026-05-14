# Strategy: 4h_Camarilla_R3S3_1dEMA34_VolumeSpike_Balanced

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.075 | +23.3% | -11.7% | 247 | KEEP |
| ETHUSDT | 0.066 | +22.0% | -15.7% | 240 | KEEP |
| SOLUSDT | 0.705 | +102.5% | -19.8% | 232 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.821 | -2.8% | -7.5% | 91 | DISCARD |
| ETHUSDT | 0.436 | +12.9% | -11.2% | 80 | KEEP |
| SOLUSDT | 0.619 | +16.6% | -7.4% | 80 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Uses Camarilla pivot levels from daily timeframe for precise institutional breakout levels
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.5x 20 EMA volume) filters false breakouts
# Discrete sizing 0.25 minimizes fee churn while targeting 75-200 total trades over 4 years
# Works in bull markets (continuation at R3/R4) and bear markets (continuation at S3/S4)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_Balanced"
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
    
    # Get 1d data for Camarilla calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior completed 1d Camarilla levels (R3, R4, S3, S4)
    # Camarilla: Range = high - low
    # R4 = close + Range * 1.1/2
    # R3 = close + Range * 1.1/4
    # S3 = close - Range * 1.1/4
    # S4 = close - Range * 1.1/2
    daily_range = high_1d - low_1d
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_r3 = close_1d + daily_range * 1.1 / 4
    camarilla_s3 = close_1d - daily_range * 1.1 / 4
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Shift to use prior completed 1d bar (avoid look-ahead)
    camarilla_r4_shifted = np.roll(camarilla_r4, 1)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_s4_shifted = np.roll(camarilla_s4, 1)
    camarilla_r4_shifted[0] = np.nan
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    camarilla_s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_shifted)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_shifted)
    
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
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below 1d EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above 1d EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 12:02
