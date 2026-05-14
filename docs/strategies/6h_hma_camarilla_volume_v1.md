# Strategy: 6h_hma_camarilla_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.283 | +34.7% | -9.4% | 166 | PASS |
| ETHUSDT | 0.472 | +50.8% | -14.8% | 148 | PASS |
| SOLUSDT | 1.012 | +157.8% | -19.2% | 123 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.069 | -5.4% | -11.7% | 60 | FAIL |
| ETHUSDT | 0.772 | +19.6% | -7.9% | 50 | PASS |
| SOLUSDT | -0.954 | -10.8% | -21.2% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_hma_camarilla_volume_v1
# Hypothesis: 6h Hull Moving Average (HMA) crossover with 1d Camarilla H3/L3 filter and volume confirmation.
# Uses 6h timeframe to balance trade frequency and responsiveness. HMA provides smooth trend signals with reduced lag,
# Camarilla H3/L3 acts as strong bias filter from daily pivot structure (only trade in direction of daily extremes),
# volume spike confirms institutional participation. Designed for 12-37 trades/year (50-150 over 4 years).
# Works in bull/bear markets: HMA captures trends with less whipsaw, Camarilla filter avoids counter-trend fakes during ranging.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_hma_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
    # WMA for full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for HMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 55:  # Need enough for HMA(55)
        return np.zeros(n)
    
    # Calculate 6h HMA(55) for trend
    close_6h = df_6h['close'].values
    hma_6h = calculate_hma(close_6h, 55)
    
    # Align 6h HMA to 6h timeframe (completed 6h candle only)
    hma_6h_aligned = align_htf_to_ltf(prices, df_6h, hma_6h)
    
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
    
    # Camarilla H3/L3 levels (stronger bias filter than H4/L4)
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (completed daily candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_6h_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h HMA
            if close[i] < hma_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h HMA
            if close[i] > hma_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 6h HMA, above 1d H3, with volume spike
            if (close[i] > hma_6h_aligned[i]) and (close[i] > h3_1d_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 6h HMA, below 1d L3, with volume spike
            elif (close[i] < hma_6h_aligned[i]) and (close[i] < l3_1d_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 05:48
