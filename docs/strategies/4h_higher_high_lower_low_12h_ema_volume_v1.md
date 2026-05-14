# Strategy: 4h_higher_high_lower_low_12h_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.155 | +9.6% | -22.6% | 137 | FAIL |
| ETHUSDT | -0.047 | +12.7% | -16.0% | 133 | FAIL |
| SOLUSDT | 0.761 | +130.3% | -25.5% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.287 | +10.7% | -11.8% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_higher_high_lower_low_12h_ema_volume_v1
# Hypothesis: 4h higher high/lower low breakout with 12h EMA(34) trend filter and 4h volume confirmation.
# 12h EMA(34) determines primary trend (only long above, short below).
# 4h breakout above recent swing high (higher high) or below recent swing low (lower low) provides entry.
# 4h volume > 1.6x 20-period average confirms institutional participation.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-50 trades/year.
# Works in bull/bear: EMA filter avoids counter-trend trades, volume ensures momentum validity.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_higher_high_lower_low_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h data for swing high/low detection and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Swing high: highest high in last 20 4h bars
    swing_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Swing low: lowest low in last 20 4h bars
    swing_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to primary timeframe
    swing_high_4h_aligned = align_htf_to_ltf(prices, df_4h, swing_high_4h)
    swing_low_4h_aligned = align_htf_to_ltf(prices, df_4h, swing_low_4h)
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(swing_high_4h_aligned[i]) or
            np.isnan(swing_low_4h_aligned[i]) or np.isnan(volume_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below swing low OR 12h EMA turns bearish (price < EMA)
            if close[i] < swing_low_4h_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above swing high OR 12h EMA turns bullish (price > EMA)
            if close[i] > swing_high_4h_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current volume > 1.6x 20-period average of 4h volume
            volume_confirmed = volume[i] > 1.6 * volume_ma_4h_aligned[i]
            
            if volume_confirmed:
                # Long entry: price breaks above swing high (higher high) AND above 12h EMA (uptrend)
                if close[i] > swing_high_4h_aligned[i] and close[i] > ema_34_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below swing low (lower low) AND below 12h EMA (downtrend)
                elif close[i] < swing_low_4h_aligned[i] and close[i] < ema_34_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 03:58
