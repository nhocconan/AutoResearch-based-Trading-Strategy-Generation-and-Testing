# Strategy: 6h_donchian_12h_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.405 | -4.7% | -19.8% | 52 | FAIL |
| ETHUSDT | 0.041 | +18.4% | -27.3% | 48 | PASS |
| SOLUSDT | 0.771 | +141.0% | -28.8% | 49 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.010 | +4.7% | -16.8% | 17 | PASS |
| SOLUSDT | -0.454 | -7.0% | -19.9% | 18 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_donchian_12h_pivot_volume_v1
# Hypothesis: 6h strategy using 12h Donchian breakout with 1d Camarilla pivot direction filter and volume confirmation.
# Enters long when price breaks above 12h Donchian(20) upper band, price is above 1d Camarilla H3 level, and volume > 1.5x 20-period average.
# Enters short when price breaks below 12h Donchian(20) lower band, price is below 1d Camarilla L3 level, and volume > 1.5x average.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via Donchian structure and pivot direction filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_12h_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data ONCE before loop for Donchian
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels for 12h (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper = rolling max of high, lower = rolling min of low
    donchian_upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (completed 12h candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # Get 1d HTF data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for daily (H3/L3 for direction filter)
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (completed daily candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 12h Donchian lower band
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 12h Donchian upper band
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h Donchian upper, above 1d H3, with volume spike
            if (close[i] > donchian_upper_aligned[i]) and (close[i] > h3_1d_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h Donchian lower, below 1d L3, with volume spike
            elif (close[i] < donchian_lower_aligned[i]) and (close[i] < l3_1d_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 05:35
