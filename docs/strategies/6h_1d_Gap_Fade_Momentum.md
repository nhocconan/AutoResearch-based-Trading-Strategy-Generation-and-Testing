# Strategy: 6h_1d_Gap_Fade_Momentum

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.658 | +45.5% | -6.3% | 158 | PASS |
| ETHUSDT | 0.126 | +25.7% | -14.9% | 138 | PASS |
| SOLUSDT | 0.064 | +21.3% | -24.0% | 129 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.257 | -4.1% | -6.7% | 67 | FAIL |
| ETHUSDT | 0.858 | +16.7% | -5.1% | 58 | PASS |
| SOLUSDT | -0.573 | -1.2% | -11.3% | 48 | FAIL |

## Code
```python
# 6h_1d_Gap_Fade_Momentum
# Hypothesis: Price gaps on daily chart often reverse on 6h timeframe due to overnight/weekend liquidity imbalances.
# In bull markets: fade down gaps (price < prior day low) with momentum confirmation.
# In bear markets: fade up gaps (price > prior day high) with momentum confirmation.
# Uses 60-period EMA on 6h for trend filter and volume spike for entry confirmation.
# Targets 15-30 trades/year with position size 0.25 to avoid fee drag.
# Works in both bull/bear by fading gaps against prevailing trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Gap_Fade_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for gap detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily gap detection: today's open vs yesterday's close
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    gap_up = daily_open > daily_close  # Gap up when open > prior close
    gap_down = daily_open < daily_close  # Gap down when open < prior close
    
    # Align daily gap signals to 6h timeframe
    gap_up_aligned = align_htf_to_ltf(prices, df_1d, gap_up.astype(float))
    gap_down_aligned = align_htf_to_ltf(prices, df_1d, gap_down.astype(float))
    
    # 60-period EMA on 6h for trend filter (responsive but smooth)
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.where(vol_ma > 0, volume / vol_ma, 1.0) > 2.0
    
    # Momentum confirmation: 6-period ROC > 0 for longs, < 0 for shorts
    roc_6 = ((close_series / close_series.shift(6)) - 1) * 100
    roc_6_values = roc_6.fillna(0).values
    mom_long = roc_6_values > 0
    mom_short = roc_6_values < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_60[i]) or np.isnan(gap_up_aligned[i]) or 
            np.isnan(gap_down_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(mom_long[i]) or np.isnan(mom_short[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below EMA60
        uptrend = close[i] > ema_60[i]
        downtrend = close[i] < ema_60[i]
        
        if position == 0:
            # Long: gap down day + volume spike + bullish momentum in uptrend
            long_condition = (gap_down_aligned[i] > 0.5) and vol_spike[i] and mom_long[i] and uptrend
            # Short: gap up day + volume spike + bearish momentum in downtrend
            short_condition = (gap_up_aligned[i] > 0.5) and vol_spike[i] and mom_short[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: gap up day or momentum turns bearish
            if (gap_up_aligned[i] > 0.5) or (not mom_long[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: gap down day or momentum turns bullish
            if (gap_down_aligned[i] > 0.5) or (not mom_short[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
```

## Last Updated
2026-05-07 21:56
