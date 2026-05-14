# Strategy: 4H_Camarilla_R3S3_1DTrend_Volume_Signal_v5

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.324 | +35.8% | -9.1% | 98 | PASS |
| ETHUSDT | 0.511 | +51.4% | -11.5% | 91 | PASS |
| SOLUSDT | 0.940 | +133.6% | -20.5% | 81 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.646 | -0.4% | -8.4% | 37 | FAIL |
| ETHUSDT | 1.608 | +35.3% | -6.4% | 28 | PASS |
| SOLUSDT | 0.594 | +14.4% | -9.2% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R3S3_1DTrend_Volume_Signal_v5
# Hypothesis: Apply a volatility filter (ATR-based) to the proven Camarilla R3/S3 breakout strategy to reduce false signals during low-volatility periods. This should decrease trade frequency and improve signal quality in both bull and bear markets by only taking trades when volatility is sufficient for meaningful moves. Target: 15-30 trades/year per symbol.

name = "4H_Camarilla_R3S3_1DTrend_Volume_Signal_v5"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from previous daily period's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    hl_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * hl_range / 2
    s3_1d = close_1d - 1.1 * hl_range / 2
    
    # Align all levels to 4h timeframe (use previous daily period's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate EMA34 for trend filter (daily)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR moving average (50-period) for volatility regime filter
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Volume spike detection: 2.0x average volume (50-period for responsiveness)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure we have ATR MA, volatility filter, and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_ma[i]) or
            vol_ma[i] == 0 or atr_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when current ATR is above its 50-period average
        vol_filter = atr[i] > atr_ma[i]
        
        if position == 0:
            # Long: price breaks above daily R3, price above daily EMA34 (uptrend), volume spike, volatility sufficient
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3, price below daily EMA34 (downtrend), volume spike, volatility sufficient
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below daily S3 (opposite level)
            if close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above daily R3 (opposite level)
            if close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 03:05
