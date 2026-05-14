# Strategy: 4h_Donchian20_Volume_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.297 | +32.1% | -9.5% | 382 | PASS |
| ETHUSDT | 0.284 | +33.5% | -7.9% | 348 | PASS |
| SOLUSDT | 0.034 | +19.3% | -21.5% | 329 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.250 | -2.6% | -7.5% | 151 | FAIL |
| ETHUSDT | 0.587 | +13.0% | -6.7% | 139 | PASS |
| SOLUSDT | 0.432 | +11.2% | -9.8% | 118 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above Donchian upper band AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND volume > 1.8x 20-period average.
Exit when price retraces 50% of the ATR from the extreme favorable price since entry.
Uses proven Donchian breakout structure with volume filter to reduce false breakouts.
Designed for low trade frequency (20-50/year) to minimize fee drag while capturing strong breakouts.
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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) on 4h for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: use high-low
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    extreme_price = 0.0  # Tracks best price since entry for trailing stop
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        atr = atr_aligned[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above upper band AND volume > 1.8x avg
            if high_price > upper and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
                extreme_price = price
            # Short: price breaks below lower band AND volume > 1.8x avg
            elif low_price < lower and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
                extreme_price = price
        
        elif position == 1:
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Exit long: price retraces 50% of ATR from extreme price
            if price < extreme_price - 0.5 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update extreme price (lowest since entry)
            if price < extreme_price:
                extreme_price = price
            # Exit short: price retraces 50% of ATR from extreme price
            if price > extreme_price + 0.5 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRTrail"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 20:47
