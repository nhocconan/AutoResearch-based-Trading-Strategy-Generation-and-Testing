# Strategy: 6h_donchian_1d_camarilla_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.494 | -5.7% | -20.6% | 71 | FAIL |
| ETHUSDT | 0.291 | +39.2% | -16.2% | 65 | PASS |
| SOLUSDT | 0.866 | +146.6% | -26.4% | 59 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.107 | +6.9% | -11.6% | 25 | PASS |
| SOLUSDT | -0.010 | +4.3% | -18.5% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_donchian_1d_camarilla_v1
# Hypothesis: 6h Donchian(20) breakout with 1d Camarilla H4/L4 filter and volume confirmation.
# Uses 6h timeframe to reduce trade frequency vs 4h strategies. Donchian provides trend following,
# Camarilla H4/L4 acts as strong bias filter (only trade in direction of daily pivot extremes),
# volume spike confirms institutional interest. Designed for 12-37 trades/year (50-150 over 4 years).
# Works in bull/bear markets: breakouts capture trends, Camarilla filter avoids counter-trend fakes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1d_camarilla_v1"
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
    
    # Get 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    donchian_upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align 6h Donchian levels to 6h timeframe (completed 6h candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_6h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_6h)
    
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
    
    # Camarilla levels for daily (H4/L4 for stronger direction filter)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (completed daily candle only)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h Donchian lower band
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h Donchian upper band
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 6h Donchian upper, above 1d H4, with volume spike
            if (close[i] > donchian_upper_aligned[i]) and (close[i] > h4_1d_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 6h Donchian lower, below 1d L4, with volume spike
            elif (close[i] < donchian_lower_aligned[i]) and (close[i] < l4_1d_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 05:42
