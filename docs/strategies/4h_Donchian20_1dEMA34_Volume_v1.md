# Strategy: 4h_Donchian20_1dEMA34_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.115 | +25.3% | -10.2% | 179 | KEEP |
| ETHUSDT | 0.024 | +20.1% | -12.3% | 174 | KEEP |
| SOLUSDT | 0.582 | +77.5% | -23.8% | 179 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.249 | -4.4% | -7.2% | 66 | DISCARD |
| ETHUSDT | 0.535 | +13.8% | -6.8% | 57 | KEEP |
| SOLUSDT | 0.430 | +12.3% | -8.4% | 54 | KEEP |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.5x average
# Exit when price crosses Donchian middle band (mean reversion) OR trend reversal (price crosses 1d EMA34)
# Uses 4h timeframe for optimal trade frequency, Donchian for structure, 1d EMA for trend filter, volume for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Donchian20_1dEMA34_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    # Align Donchian levels to 15m timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper Donchian AND close > 1d EMA34 AND volume confirmation
            if close[i] > highest_20_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower Donchian AND close < 1d EMA34 AND volume confirmation
            elif close[i] < lowest_20_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle Donchian OR trend reversal (close < 1d EMA34)
            if close[i] < middle_20_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle Donchian OR trend reversal (close > 1d EMA34)
            if close[i] > middle_20_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 14:26
