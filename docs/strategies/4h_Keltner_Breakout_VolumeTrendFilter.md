# Strategy: 4h_Keltner_Breakout_VolumeTrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.082 | +16.5% | -11.7% | 102 | FAIL |
| ETHUSDT | 0.381 | +41.6% | -10.6% | 96 | PASS |
| SOLUSDT | 0.566 | +68.0% | -23.6% | 84 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.435 | +12.0% | -7.8% | 37 | PASS |
| SOLUSDT | 0.191 | +8.3% | -13.6% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_VolumeTrendFilter
# Hypothesis: Keltner Channel breakouts with volume confirmation and EMA trend filter capture sustained moves in both bull and bear markets.
# The 4h timeframe reduces trade frequency while Keltner Channels (ATR-based) adapt to volatility better than fixed bands.
# Works in bull markets by catching breakouts above upper channel; in bear markets by catching breakdowns below lower channel.
# Volume filter ensures institutional participation; EMA filter avoids counter-trend trades.

name = "4h_Keltner_Breakout_VolumeTrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 trend filter on daily timeframe
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate Keltner Channel on 4h data (20-period EMA, 2*ATR)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Volume filter: volume > 1.8x 20-period EMA (more selective than SMA)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner + above daily EMA20 (uptrend) + volume confirmation
            if close[i] > kc_upper[i] and close[i] > ema20_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + below daily EMA20 (downtrend) + volume confirmation
            elif close[i] < kc_lower[i] and close[i] < ema20_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below EMA20 (trend change) or below lower Keltner (mean reversion)
            if close[i] < ema20[i] or close[i] < kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above EMA20 (trend change) or above upper Keltner (mean reversion)
            if close[i] > ema20[i] or close[i] > kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 01:22
