# Strategy: 12h_camarilla_pivot_1d_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.350 | -0.6% | -14.9% | 184 | FAIL |
| ETHUSDT | -0.025 | +14.7% | -17.5% | 181 | FAIL |
| SOLUSDT | 0.555 | +85.4% | -24.1% | 155 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.049 | +5.6% | -15.2% | 53 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe provide strong intraday support/resistance.
In ranging markets (Choppiness > 61.8), price tends to revert to these levels.
In trending markets (Choppiness < 38.2), breakouts of S3/R3 levels offer continuation trades.
Volume confirmation filters false breakouts. Designed for 12h timeframe to target 12-37 trades/year.
Works in both bull and bear markets by adapting to market regime via Choppiness filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (high values = ranging, low = trending)"""
    atr = np.zeros(len(high))
    atr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing (alpha = 1/period)
    for i in range(1, len(atr)):
        atr[i] = (atr[i-1] * (period-1) + atr[i]) / period
    
    # Calculate highest high and lowest low over period
    hh = np.zeros(len(high))
    ll = np.zeros(len(high))
    hh[0] = high[0]
    ll[0] = low[0]
    for i in range(1, len(high)):
        hh[i] = max(high[i], hh[i-1])
        ll[i] = min(low[i], ll[i-1])
    
    # Choppiness formula
    chop = np.zeros(len(high))
    for i in range(period-1, len(high)):
        sum_atr = 0
        for j in range(i-period+1, i+1):
            sum_atr += atr[j]
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(sum_atr / (hh[i] - ll[i])) / np.log10(period)
        else:
            chop[i] = 50  # neutral when no range
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily data for Camarilla pivots and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas (based on previous day's range)
    S1 = close_1d - (high_1d - low_1d) * 1.0833 / 2
    S2 = close_1d - (high_1d - low_1d) * 1.1666 / 2
    S3 = close_1d - (high_1d - low_1d) * 1.2500 / 2
    R1 = close_1d + (high_1d - low_1d) * 1.0833 / 2
    R2 = close_1d + (high_1d - low_1d) * 1.1666 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.2500 / 2
    
    # Align daily data to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # Calculate Choppiness on daily data for regime filter
    chop = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Regime filters
        ranging = chop_aligned[i] > 61.8  # Chop > 61.8 = ranging (mean revert)
        trending = chop_aligned[i] < 38.2  # Chop < 38.2 = trending (breakout)
        
        if position == 1:  # Long position
            # Exit: price reaches S1 (support) or chop shifts to ranging
            if close[i] <= S1_aligned[i] or (ranging and chop_aligned[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 (resistance) or chop shifts to ranging
            if close[i] >= R1_aligned[i] or (ranging and chop_aligned[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long in ranging market: price at S3 with volume confirmation
            if ranging and close[i] <= S3_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short in ranging market: price at R3 with volume confirmation
            elif ranging and close[i] >= R3_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
            # Long in trending market: break above R3 with volume
            elif trending and close[i] > R3_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short in trending market: break below S3 with volume
            elif trending and close[i] < S3_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 20:32
