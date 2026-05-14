# Strategy: 4h_PriceAction_12hTrend_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.046 | +17.6% | -13.3% | 260 | FAIL |
| ETHUSDT | 0.174 | +29.1% | -10.8% | 259 | PASS |
| SOLUSDT | 0.564 | +75.2% | -25.4% | 248 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.190 | +8.3% | -9.6% | 82 | PASS |
| SOLUSDT | 0.130 | +7.4% | -10.6% | 83 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_PriceAction_12hTrend_VolumeFilter
Hypothesis: Price action breakout from 4h support/resistance with 12h trend filter and volume confirmation.
Long when price breaks above 4h resistance with volume > 1.5x average and 12h uptrend.
Short when price breaks below 4h support with volume > 1.5x average and 12h downtrend.
Exit when price returns to 4h midpoint or trend reverses.
Designed to capture strong momentum moves while avoiding false breakouts in low volume.
Target: 25-40 trades/year to minimize fee drag while capturing major moves.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_12h_period = 34
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_12h_period:
        ema_12h[ema_12h_period - 1] = np.mean(close_12h[:ema_12h_period])
        multiplier = 2 / (ema_12h_period + 1)
        for i in range(ema_12h_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Calculate 4h resistance/support (14-period high/low)
    resistance_14 = np.full(n, np.nan)
    support_14 = np.full(n, np.nan)
    for i in range(14, n):
        resistance_14[i] = np.max(high[i-14:i])
        support_14[i] = np.min(low[i-14:i])
    
    # Calculate 4h volume average (14-period)
    vol_ma_14 = np.full(n, np.nan)
    for i in range(14, n):
        vol_ma_14[i] = np.mean(volume[i-14:i])
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(14, 34)  # 4h range needs 14, EMA needs 34
    
    for i in range(start_idx, n):
        if (np.isnan(resistance_14[i]) or
            np.isnan(support_14[i]) or
            np.isnan(vol_ma_14[i]) or
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_14[i] if vol_ma_14[i] > 0 else 0
        
        # Trend filter: 12h EMA34
        uptrend = price > ema_12h_aligned[i]
        downtrend = price < ema_12h_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: break above 4h resistance with volume and uptrend
            if uptrend and volume_confirmation and price > resistance_14[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 4h support with volume and downtrend
            elif downtrend and volume_confirmation and price < support_14[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: return to 4h midpoint or trend reversal
            midpoint = (resistance_14[i] + support_14[i]) / 2
            if price < midpoint or price <= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: return to 4h midpoint or trend reversal
            midpoint = (resistance_14[i] + support_14[i]) / 2
            if price > midpoint or price >= ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_PriceAction_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 14:59
