# Strategy: 6h_Donchian20_12hEMA50_VolumeConfirm_Balanced

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.383 | -0.5% | -19.1% | 85 | FAIL |
| ETHUSDT | 0.059 | +20.8% | -12.5% | 79 | PASS |
| SOLUSDT | 0.965 | +168.2% | -21.6% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.529 | +15.7% | -9.0% | 26 | PASS |
| SOLUSDT | -0.546 | -6.3% | -20.7% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses 6h Donchian channel (20-bar high/low) for structure - captures momentum bursts
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.8x average volume) - balanced to reduce overtrading
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets (continuation at upper band) and bear markets (continuation at lower band)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "6h_Donchian20_12hEMA50_VolumeConfirm_Balanced"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h Donchian(20) channels from prior completed 6h bar
    # Upper band = highest high over past 20 periods
    # Lower band = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 6h bar (no look-ahead)
    upper_band_shifted = np.roll(upper_band, 1)
    lower_band_shifted = np.roll(lower_band, 1)
    upper_band_shifted[0] = np.nan
    lower_band_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_band_shifted[i]) or np.isnan(lower_band_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND price > 12h EMA50 AND volume spike
            if close[i] > upper_band_shifted[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND price < 12h EMA50 AND volume spike
            elif close[i] < lower_band_shifted[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower band OR price crosses below 12h EMA50
            if close[i] < lower_band_shifted[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper band OR price crosses above 12h EMA50
            if close[i] > upper_band_shifted[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 10:14
