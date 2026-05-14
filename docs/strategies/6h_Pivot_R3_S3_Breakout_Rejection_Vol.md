# Strategy: 6h_Pivot_R3_S3_Breakout_Rejection_Vol

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.386 | +10.9% | -11.6% | 227 | FAIL |
| ETHUSDT | 0.255 | +30.6% | -7.0% | 203 | PASS |
| SOLUSDT | 0.304 | +39.1% | -14.3% | 188 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.140 | +7.3% | -5.5% | 83 | PASS |
| SOLUSDT | 0.025 | +6.2% | -3.7% | 66 | PASS |

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Shift to use previous day's pivots (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    
    # Align daily pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_prev)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_prev)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_prev)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_prev)
    
    # Volume confirmation: current volume > 1.5 * 24-period average (6h * 4 = 24h)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 24  # Need volume MA24 and ATR MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma24[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 24-period average
        volume_filter = volume[i] > (1.5 * volume_ma24[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and volatility (strong breakout)
            if close[i] > r3_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and volatility (strong breakdown)
            elif close[i] < s3_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S1 and moves back above it (bullish rejection)
            elif close[i] > s1_6h[i] and low[i] < s1_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R1 and moves back below it (bearish rejection)
            elif close[i] < r1_6h[i] and high[i] > r1_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R3_S3_Breakout_Rejection_Vol"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 08:51
