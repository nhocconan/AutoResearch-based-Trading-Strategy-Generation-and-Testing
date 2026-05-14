# Strategy: 4h_Donchian20_12hEMA50_VolumeConfirm_Balanced_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.045 | +16.8% | -15.1% | 96 | FAIL |
| ETHUSDT | 0.243 | +34.5% | -16.1% | 89 | PASS |
| SOLUSDT | 0.597 | +86.4% | -37.8% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.029 | +5.5% | -10.4% | 34 | PASS |
| SOLUSDT | -0.120 | +2.5% | -15.5% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>2.0x 20 EMA volume)
# Uses 4h Donchian channel breakouts for structure - captures strong momentum
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>2.0x average volume) - tighter to reduce trades to target
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "4h_Donchian20_12hEMA50_VolumeConfirm_Balanced_v2"
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
    
    # Calculate 4h Donchian(20) channels from prior completed 4h bar
    # For 4h timeframe, we use the previous completed 4h bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Donchian upper channel: highest high of last 20 periods
    upper_channel = np.full(n, np.nan)
    # Donchian lower channel: lowest low of last 20 periods
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_channel[i] = np.max(prev_high[i-19:i+1])
        lower_channel[i] = np.min(prev_low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND price > 12h EMA50 AND volume spike
            if close[i] > upper_channel[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND price < 12h EMA50 AND volume spike
            elif close[i] < lower_channel[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower channel OR price crosses below 12h EMA50
            if close[i] < lower_channel[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper channel OR price crosses above 12h EMA50
            if close[i] > upper_channel[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 10:35
