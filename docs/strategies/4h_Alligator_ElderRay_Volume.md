# Strategy: 4h_Alligator_ElderRay_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.111 | +14.2% | -11.5% | 234 | DISCARD |
| ETHUSDT | 0.536 | +55.8% | -11.9% | 224 | KEEP |
| SOLUSDT | 0.362 | +51.7% | -19.2% | 198 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.210 | +8.8% | -13.0% | 71 | KEEP |
| SOLUSDT | -0.457 | -2.4% | -19.9% | 58 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (13,8,5 SMAs) + Elder Ray (13-bar EMA bull/bear power) + volume spike (2x 20-period avg).
# Alligator defines trend direction (jaws/teeth/lips alignment), Elder Ray confirms strength, volume filters weak moves.
# Designed for 4h timeframe to capture medium-term trends in both bull and bear markets with low trade frequency.
# Entry: Bullish Alligator alignment + positive Elder Ray power + volume spike.
# Exit: Bearish Alligator alignment OR negative Elder Ray power.
# Uses strict conditions to limit trades (~20-30/year) and avoid overtrading.
name = "4h_Alligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # 13-period smoothed by 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # 8-period smoothed by 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values  # 5-period smoothed by 3
    
    # Elder Ray Power: 13-bar EMA of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish Alligator (lips > teeth > jaws) + positive bull power + volume spike
            if (lips[i] > teeth[i] > jaws[i] and 
                bull_power[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator (jaws > teeth > lips) + negative bear power + volume spike
            elif (jaws[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish Alligator OR negative bull power
            if (jaws[i] > teeth[i] > lips[i]) or (bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish Alligator OR positive bear power
            if (lips[i] > teeth[i] > jaws[i]) or (bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 19:37
