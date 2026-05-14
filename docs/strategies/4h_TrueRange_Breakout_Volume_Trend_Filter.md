# Strategy: 4h_TrueRange_Breakout_Volume_Trend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.088 | +23.9% | -17.7% | 239 | PASS |
| ETHUSDT | 0.222 | +33.1% | -14.9% | 229 | PASS |
| SOLUSDT | 0.396 | +56.3% | -31.7% | 224 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.317 | -7.3% | -9.6% | 85 | FAIL |
| ETHUSDT | 0.106 | +7.0% | -9.8% | 79 | PASS |
| SOLUSDT | 0.082 | +6.5% | -13.4% | 76 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_TrueRange_Breakout_Volume_Trend_Filter
Hypothesis: Price breaks above/below the True Range (ATR-based) channel with volume confirmation and EMA trend filter.
Uses ATR(14) to define dynamic breakout levels: Upper = SMA(20) + 1.5*ATR(14), Lower = SMA(20) - 1.5*ATR(14).
Requires volume > 1.5x 20-period average and EMA20 trend alignment.
Designed to capture volatility expansion moves in both bull and bear markets with tight entry conditions.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility-based channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # SMA(20) for mean line
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Dynamic breakout channels: SMA ± 1.5*ATR
    upper_channel = sma_20 + 1.5 * atr
    lower_channel = sma_20 - 1.5 * atr
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # EMA20 trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for SMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        vol_ok = volume_filter[i]
        ema20 = ema_20[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume in uptrend
            if price > upper and vol_ok and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume in downtrend
            elif price < lower and vol_ok and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to middle (SMA) or trend reverses
            if price < sma_20[i] or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to middle (SMA) or trend reverses
            if price > sma_20[i] or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TrueRange_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 03:45
