# Strategy: 6h_AroonOscillator_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.381 | +0.9% | -17.5% | 86 | DISCARD |
| ETHUSDT | 0.005 | +17.6% | -13.8% | 78 | KEEP |
| SOLUSDT | 0.617 | +91.3% | -29.2% | 77 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.075 | +6.4% | -11.8% | 29 | KEEP |
| SOLUSDT | -0.389 | -3.0% | -16.7% | 29 | DISCARD |

## Code
```python
#!/usr/bin/env python3
# 6h_AroonOscillator_12hTrend_Volume
# Hypothesis: Uses Aroon Oscillator (25-period) to detect trend strength and direction, filtered by 12h EMA50 trend and volume confirmation.
# Aroon Oscillator ranges from -100 to +100, with values above +50 indicating strong uptrend and below -50 indicating strong downtrend.
# This helps capture sustained trends while avoiding choppy markets. Works in both bull and bear markets by only trading in the direction of the 12h trend.
# Target: 15-30 trades/year to stay within optimal frequency range and minimize fee drag.

name = "6h_AroonOscillator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter and Aroon calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Aroon Oscillator (25-period) on 12h data
    # Aroon Up = ((25 - periods since 25-period high) / 25) * 100
    # Aroon Down = ((25 - periods since 25-period low) / 25) * 100
    # Aroon Oscillator = Aroon Up - Aroon Down
    period = 25
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    aroon_up = np.full(len(close_12h), np.nan)
    aroon_down = np.full(len(close_12h), np.nan)
    
    for i in range(period - 1, len(close_12h)):
        # Find highest high in last 'period' periods
        period_high_idx = np.argmax(high_12h[i - period + 1:i + 1])
        periods_since_high = period - 1 - period_high_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        
        # Find lowest low in last 'period' periods
        period_low_idx = np.argmin(low_12h[i - period + 1:i + 1])
        periods_since_low = period - 1 - period_low_idx
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # Range: -100 to +100
    
    # Align Aroon Oscillator to 6h timeframe
    aroon_osc_6h = align_htf_to_ltf(prices, df_12h, aroon_osc)
    
    # Calculate volume spike on 6h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(aroon_osc_6h[i]) or np.isnan(ema_50_12h_6h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Aroon Oscillator > 50 (strong uptrend) + above 12h EMA50 + volume spike
            if aroon_osc_6h[i] > 50 and close[i] > ema_50_12h_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Aroon Oscillator < -50 (strong downtrend) + below 12h EMA50 + volume spike
            elif aroon_osc_6h[i] < -50 and close[i] < ema_50_12h_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Aroon Oscillator falls below 0 (trend weakening) or price closes below 12h EMA50
            if aroon_osc_6h[i] < 0 or close[i] < ema_50_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Aroon Oscillator rises above 0 (trend weakening) or price closes above 12h EMA50
            if aroon_osc_6h[i] > 0 or close[i] > ema_50_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 00:24
