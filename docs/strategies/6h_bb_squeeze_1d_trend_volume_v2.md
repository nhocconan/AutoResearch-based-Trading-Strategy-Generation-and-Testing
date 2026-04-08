# Strategy: 6h_bb_squeeze_1d_trend_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.091 | +17.8% | -12.9% | 55 | FAIL |
| ETHUSDT | -0.173 | +13.8% | -13.5% | 56 | FAIL |
| SOLUSDT | 0.305 | +38.4% | -17.7% | 52 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.268 | +9.5% | -9.4% | 22 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Squeeze + 1-day Trend + Volume Confirmation.
In bull market (1d close > 1d EMA50): long when Bollinger Band width contracts then expands with upward breakout.
In bear market (1d close < 1d EMA50): short when Bollinger Band width contracts then expands with downward breakout.
Volume must be above 20-period average to confirm.
Uses 6h Bollinger Bands for squeeze detection, 1d for trend filter.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_squeeze_1d_trend_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === 6H BOLLINGER BANDS (LTF) ===
    bb_length = 20
    bb_mult = 2.0
    bb_basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    bb_width = bb_upper - bb_lower
    
    # Bollinger Squeeze detection: width < 50-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < bb_width_ma  # True when in squeeze
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(bb_width_ma[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below BB middle OR trend turns bearish
            if close[i] < bb_basis[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above BB middle OR trend turns bullish
            if close[i] > bb_basis[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Look for breakout after squeeze
            # Must be coming out of squeeze (previous bar in squeeze, current bar not)
            if i > 0 and squeeze_condition[i-1] and not squeeze_condition[i]:
                if bull_trend:
                    # In bull market: long on upward breakout above BB upper
                    if close[i] > bb_upper[i]:
                        position = 1
                        signals[i] = 0.25
                else:
                    # In bear market: short on downward breakout below BB lower
                    if close[i] < bb_lower[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 21:42
