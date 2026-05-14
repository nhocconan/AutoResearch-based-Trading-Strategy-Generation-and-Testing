# Strategy: 6h_ElderRay_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.279 | +2.6% | -19.3% | 284 | FAIL |
| ETHUSDT | 0.181 | +30.3% | -19.0% | 281 | PASS |
| SOLUSDT | 1.431 | +354.8% | -27.7% | 241 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.087 | +6.4% | -11.2% | 101 | PASS |
| SOLUSDT | -0.182 | +0.5% | -12.3% | 94 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13, indicating trend strength.
# 12h EMA50 provides higher-timeframe trend bias to avoid counter-trend trades.
# Volume confirmation ensures trades have participation.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets via trend-filtered momentum.

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Get 6h data for volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirmed = volume[i] > (1.3 * vol_ema_20[i])
        
        # 12h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_12h_aligned[i]
        bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) + volume confirmation + bullish 12h trend
            if (bull_power[i] > 0 and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + volume confirmation + bearish 12h trend
            elif (bear_power[i] < 0 and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (bulls lose control) OR 12h trend turns bearish
            if bull_power[i] <= 0 or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (bears lose control) OR 12h trend turns bullish
            if bear_power[i] >= 0 or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 03:53
