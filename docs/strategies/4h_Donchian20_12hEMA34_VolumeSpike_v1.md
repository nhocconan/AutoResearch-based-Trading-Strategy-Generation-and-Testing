# Strategy: 4h_Donchian20_12hEMA34_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.208 | +11.4% | -15.0% | 158 | DISCARD |
| ETHUSDT | 0.612 | +58.0% | -12.9% | 141 | KEEP |
| SOLUSDT | 0.406 | +53.7% | -27.1% | 134 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.469 | +12.7% | -6.6% | 53 | KEEP |
| SOLUSDT | 0.013 | +5.5% | -14.7% | 44 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume spike confirmation
# Long when price breaks above 20-period 4h Donchian high AND 12h close > 12h EMA34 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 20-period 4h Donchian low AND 12h close < 12h EMA34 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 4h Donchian midpoint (mean of 20-period high and low)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide robust breakout levels based on price extremes
# 12h EMA34 filters for higher timeframe trend alignment
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 12h trend

name = "4h_Donchian20_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels and 12h EMA34 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 20 or len(df_12h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels for 4h timeframe
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Calculate 12h EMA34 trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed bars)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema34_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema34_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests midpoint from above
            if close[i] <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests midpoint from below
            if close[i] >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 20:22
