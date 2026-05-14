# Strategy: 4h_Camarilla_Volume_Chop_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.211 | +16.3% | -7.3% | 106 | FAIL |
| ETHUSDT | 0.247 | +29.2% | -5.6% | 98 | PASS |
| SOLUSDT | -0.322 | +4.7% | -19.1% | 76 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.705 | +12.8% | -5.8% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot + Volume Spike + Choppiness Regime Filter
# Uses institutional pivot points from 1d data with volume confirmation
# Choppiness regime filter avoids false breakouts in sideways markets
# Works in bull/bear by only taking breakouts in trending regimes
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    # L4 = C - ((H - L) * 1.1 / 2)
    # H4 = C + ((H - L) * 1.1 / 2)
    camarilla_h4_1d = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Choppiness index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR) / (HHV - LLV)) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = hh - ll
    chop = np.full_like(close, 50.0, dtype=float)  # default to neutral
    mask = denominator != 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / denominator[mask]) / np.log10(14)
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for chop calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Regime filter: only trade in trending markets (Chop < 38.2)
        if chop[i] >= 38.2:
            # In ranging choppy market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H4 with volume filter
            if price > camarilla_h4_aligned[i] and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Camarilla L4 with volume filter
            elif price < camarilla_l4_aligned[i] and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla L4
            if price < camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla H4
            if price > camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 01:03
