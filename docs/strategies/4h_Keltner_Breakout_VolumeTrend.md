# Strategy: 4h_Keltner_Breakout_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.273 | +35.5% | -11.6% | 96 | PASS |
| ETHUSDT | 0.292 | +39.7% | -15.5% | 96 | PASS |
| SOLUSDT | 0.940 | +172.7% | -20.5% | 88 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.834 | -4.2% | -8.0% | 35 | FAIL |
| ETHUSDT | 0.128 | +7.3% | -9.8% | 33 | PASS |
| SOLUSDT | 0.268 | +10.4% | -10.4% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_VolumeTrend
# Hypothesis: 4h chart strategy using Keltner Channel breakout with volume confirmation and EMA50 trend filter.
# Keltner Channel (ATR-based) provides volatility-adjusted breakout levels.
# Volume > 1.5x average confirms breakout strength. EMA50 trend filter ensures alignment with higher timeframe trend.
# Designed to work in both bull and bear markets by combining volatility breakout with trend and volume filters.
# Target: 25-40 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "4h"
name = "4h_Keltner_Breakout_VolumeTrend"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for Keltner Channel (20-period ATR, 2x multiplier)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: 20-period EMA of typical price ± 2*ATR
    typical_price = (high + low + close) / 3
    ema_tp = pd.Series(typical_price).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_tp + 2 * atr
    keltner_lower = ema_tp - 2 * atr
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 6)  # Ensure we have EMA50, ATR, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner Upper with volume, and 1d trend is bullish (price > EMA50)
            if (high[i] > keltner_upper[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner Lower with volume, and 1d trend is bearish (price < EMA50)
            elif (low[i] < keltner_lower[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Keltner Lower (reversal signal)
            if low[i] < keltner_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Keltner Upper (reversal signal)
            if high[i] > keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 02:19
