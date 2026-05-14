# Strategy: 6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.619 | +44.8% | -6.1% | 147 | PASS |
| ETHUSDT | 0.210 | +29.7% | -14.2% | 120 | PASS |
| SOLUSDT | 0.424 | +51.7% | -19.6% | 113 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.327 | +3.3% | -4.0% | 53 | FAIL |
| ETHUSDT | 0.656 | +14.5% | -5.4% | 47 | PASS |
| SOLUSDT | -0.581 | -1.8% | -10.0% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via aligned SMAs.
- 1d EMA50 provides higher-timeframe trend filter to avoid counter-trend trades.
- Volume confirmation (>2.0x 20-bar average) ensures institutional participation.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 60-120 total over 4 years (15-30/year) to minimize fee drag.
- Works in bull/bear markets via 1d trend filter and Alligator's trend-following nature.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator SMAs (6h timeframe)
    # JAW: 13-period SMMA, TEETH: 8-period SMMA, LIPS: 5-period SMMA
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20) + 1  # Need enough for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Alligator bullish alignment: Lips > Teeth > Jaw (trending up)
                # Plus 1d EMA50 filter: price above 1d EMA for longs
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Alligator bearish alignment: Jaw > Teeth > Lips (trending down)
                # Plus 1d EMA50 filter: price below 1d EMA for shorts
                elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR price crosses below 1d EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR price crosses above 1d EMA50
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 01:52
