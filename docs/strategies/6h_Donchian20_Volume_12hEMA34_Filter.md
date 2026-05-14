# Strategy: 6h_Donchian20_Volume_12hEMA34_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.288 | +1.1% | -14.0% | 122 | FAIL |
| ETHUSDT | 0.168 | +29.3% | -12.0% | 112 | PASS |
| SOLUSDT | 1.028 | +218.9% | -30.9% | 113 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.513 | +15.9% | -8.2% | 37 | PASS |
| SOLUSDT | 0.091 | +6.3% | -16.7% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND price > 12h EMA34.
Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND price < 12h EMA34.
Exit when price crosses the 12h EMA34 in opposite direction.
Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Get 12h data for EMA34 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels on 6h timeframe (20-period)
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    donchian_upper = high_6h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_6h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_34 = ema_34_12h_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above upper band AND volume > 1.5x avg AND price > 12h EMA34 (bullish trend)
            if high_price > upper and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume > 1.5x avg AND price < 12h EMA34 (bearish trend)
            elif low_price < lower and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h EMA34
            if price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA34
            if price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Volume_12hEMA34_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 20:40
