# Strategy: 4h_1d_ema_trend_volume_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.278 | +3.5% | -16.8% | 400 | FAIL |
| ETHUSDT | 0.340 | +43.7% | -13.2% | 379 | PASS |
| SOLUSDT | 0.930 | +159.9% | -30.7% | 321 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.283 | +10.3% | -9.7% | 129 | PASS |
| SOLUSDT | -0.158 | +1.5% | -18.4% | 122 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_ema_trend_volume_v4
# Hypothesis: Use 10-period EMA on 1d for trend direction, 21-period EMA on 4h for entry timing, and volume confirmation for institutional participation.
# The strategy is long-only in bull markets (price above 1d EMA10) and short-only in bear markets (price below 1d EMA10).
# Works in bull markets (trend continuation) and bear markets (counter-trend bounces from extremes).
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) by requiring EMA alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_trend_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA (entry signals)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA (21-period) for entry timing
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d EMA (10-period) for trend filter
    close_1d = df_1d['close'].values
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (1 day in 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_10_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on 1d EMA
        bull_market = close[i] > ema_10_1d_aligned[i]
        bear_market = close[i] < ema_10_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA21 or trend turns bearish
            if close[i] < ema_21_4h_aligned[i] or bear_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA21 or trend turns bullish
            if close[i] > ema_21_4h_aligned[i] or bull_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above 4h EMA21 in bull market with volume
            if bull_market and close[i] > ema_21_4h_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below 4h EMA21 in bear market with volume
            elif bear_market and close[i] < ema_21_4h_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:11
