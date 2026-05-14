# Strategy: 4h_Donchian20_1dEMA34_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.278 | +35.2% | -16.3% | 97 | PASS |
| ETHUSDT | 0.106 | +24.6% | -12.3% | 106 | PASS |
| SOLUSDT | 0.681 | +105.8% | -27.6% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.130 | -5.9% | -10.0% | 44 | FAIL |
| ETHUSDT | 0.079 | +6.4% | -12.4% | 35 | PASS |
| SOLUSDT | 0.262 | +10.0% | -12.6% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses 4h Donchian channel breakouts for structure - captures strong momentum bursts
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.8x average volume) - tighter to reduce trades
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "4h_Donchian20_1dEMA34_VolumeConfirm"
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
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period) from prior completed 4h bar
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper channel: 20-period high
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    upper_channel_shifted = np.roll(upper_channel, 1)
    lower_channel_shifted = np.roll(lower_channel, 1)
    upper_channel_shifted[0] = np.nan
    lower_channel_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_channel_shifted[i]) or np.isnan(lower_channel_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND price > 1d EMA34 AND volume spike
            if close[i] > upper_channel_shifted[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND price < 1d EMA34 AND volume spike
            elif close[i] < lower_channel_shifted[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower channel OR price crosses below 1d EMA34
            if close[i] < lower_channel_shifted[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper channel OR price crosses above 1d EMA34
            if close[i] > upper_channel_shifted[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 10:22
