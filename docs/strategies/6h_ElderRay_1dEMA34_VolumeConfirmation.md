# Strategy: 6h_ElderRay_1dEMA34_VolumeConfirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.416 | +34.8% | -7.3% | 346 | PASS |
| ETHUSDT | 0.093 | +24.2% | -10.2% | 294 | PASS |
| SOLUSDT | 0.298 | +36.3% | -17.8% | 256 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.325 | +3.7% | -3.3% | 130 | FAIL |
| ETHUSDT | 0.589 | +12.3% | -5.6% | 117 | PASS |
| SOLUSDT | 1.697 | +28.2% | -5.0% | 99 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bullish/bearish power relative to EMA. Long when bull power > 0
# and price above 1d EMA, short when bear power < 0 and price below 1d EMA.
# Volume > 2x average confirms conviction. Works in both bull and bear markets by
# aligning with daily trend direction, avoiding counter-trend whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 13-period EMA for Elder Ray calculation (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 + price above 1d EMA + volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + price below 1d EMA + volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Power reverses or price crosses 1d EMA
            if position == 1:
                # Exit long: Bear Power >= 0 or price crosses below 1d EMA
                if (bear_power[i] >= 0 or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bull Power <= 0 or price crosses above 1d EMA
                if (bull_power[i] <= 0 or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 10:33
