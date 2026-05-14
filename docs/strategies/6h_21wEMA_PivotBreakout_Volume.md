# Strategy: 6h_21wEMA_PivotBreakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.087 | +18.6% | -7.8% | 77 | FAIL |
| ETHUSDT | 0.495 | +39.1% | -5.5% | 55 | PASS |
| SOLUSDT | 0.783 | +80.8% | -16.2% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.018 | +6.1% | -5.9% | 21 | PASS |
| SOLUSDT | -1.667 | -9.1% | -13.7% | 16 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA to 6h timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    
    # Align daily pivot levels to 6h timeframe
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_prev)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need EMA21, R2/S2, volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or
            np.isnan(ema21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        # Weekly trend filter: price above/below weekly EMA21
        trend_up = close[i] > ema21_1w_aligned[i]
        trend_down = close[i] < ema21_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume, volatility AND weekly uptrend
            if close[i] > r2_6h[i] and volume_filter and volatility_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume, volatility AND weekly downtrend
            elif close[i] < s2_6h[i] and volume_filter and volatility_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly EMA21 or volatility drops
            if close[i] < ema21_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly EMA21 or volatility drops
            if close[i] > ema21_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_21wEMA_PivotBreakout_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 09:20
