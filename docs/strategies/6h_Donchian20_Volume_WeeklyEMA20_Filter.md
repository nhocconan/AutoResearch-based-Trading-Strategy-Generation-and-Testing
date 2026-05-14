# Strategy: 6h_Donchian20_Volume_WeeklyEMA20_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.190 | +27.5% | -6.7% | 70 | PASS |
| ETHUSDT | 0.223 | +30.1% | -9.6% | 58 | PASS |
| SOLUSDT | 0.379 | +47.0% | -22.8% | 59 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.225 | -5.8% | -6.9% | 22 | FAIL |
| ETHUSDT | 0.334 | +9.3% | -7.5% | 17 | PASS |
| SOLUSDT | -0.486 | +1.4% | -8.5% | 13 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with volume spike and weekly EMA20 trend filter.
Long when price breaks above Donchian upper band AND volume > 2.0x 20-period average AND price > weekly EMA20.
Short when price breaks below Donchian lower band AND volume > 2.0x 20-period average AND price < weekly EMA20.
Exit when price reverts to Donchian midpoint.
Uses 6h for price/volume/Donchian, weekly for EMA20 trend filter to avoid whipsaw in ranging markets.
Targets 50-150 total trades over 4 years (12-37/year). Donchian channels provide clear breakout levels,
volume confirmation reduces fakeouts, weekly EMA ensures we trade with the higher timeframe trend.
Works in bull markets (captures uptrends with bullish weekly EMA) and bear markets (captures downtrends with bearish weekly EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels on 6h timeframe (20-period)
    high_series = pd.Series(high_6h)
    low_series = pd.Series(low_6h)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume average (20-period) on 6h
    volume_series = pd.Series(volume_6h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly timeframe
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 6h Donchian channels, volume MA, and weekly EMA20 to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_20 = ema_20_1w_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 2.0x avg AND price > weekly EMA20 (bullish trend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_20:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 2.0x avg AND price < weekly EMA20 (bearish trend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Volume_WeeklyEMA20_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 20:30
