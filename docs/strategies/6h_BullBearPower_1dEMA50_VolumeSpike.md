# Strategy: 6h_BullBearPower_1dEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.077 | +23.4% | -8.9% | 185 | PASS |
| ETHUSDT | 0.019 | +19.4% | -17.2% | 165 | PASS |
| SOLUSDT | 1.062 | +177.0% | -21.1% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.693 | -1.7% | -6.5% | 72 | FAIL |
| ETHUSDT | 0.196 | +8.6% | -8.9% | 58 | PASS |
| SOLUSDT | -0.226 | +0.7% | -14.5% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bull/Bear Power (Elder Ray) with 1d EMA50 filter and volume spike confirmation.
# Bull/Bear Power = (High/Low) - EMA13 measures trend strength.
# 1d EMA50 filter ensures we trade only in the direction of the daily trend.
# Volume spike (>2x 20-period average) confirms conviction.
# Works in bull markets (bull power positive with uptrend) and bear markets (bear power negative with downtrend).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_BullBearPower_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Bull/Bear Power
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: bull power positive AND uptrend AND volume spike
            if bull_power[i] > 0 and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative AND downtrend AND volume spike
            elif bear_power[i] < 0 and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative OR trend reverses
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive OR trend reverses
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-18 21:28
