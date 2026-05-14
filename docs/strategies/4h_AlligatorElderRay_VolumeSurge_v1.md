# Strategy: 4h_AlligatorElderRay_VolumeSurge_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.201 | +28.8% | -6.6% | 264 | PASS |
| ETHUSDT | 0.065 | +22.7% | -11.1% | 256 | PASS |
| SOLUSDT | 0.365 | +48.3% | -16.0% | 234 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.889 | -1.6% | -6.9% | 95 | FAIL |
| ETHUSDT | 0.291 | +9.7% | -10.7% | 95 | PASS |
| SOLUSDT | -0.104 | +4.0% | -9.3% | 84 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1d EMA34 for daily trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Williams Alligator
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Jaw (blue): 13-period SMMA, 8 bars ahead
    sma13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma13, 8)
    jaw[:8] = np.nan
    
    # Teeth (red): 8-period SMMA, 5 bars ahead
    sma8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma8, 5)
    teeth[:5] = np.nan
    
    # Lips (green): 5-period SMMA, 3 bars ahead
    sma5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma5, 3)
    lips[:3] = np.nan
    
    # Alligator alignment: bullish when lips > teeth > jaw
    bullish_align = (lips > teeth) & (teeth > jaw)
    # Bearish alignment: lips < teeth < jaw
    bearish_align = (lips < teeth) & (teeth < jaw)
    
    # Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema1d = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Bullish alignment + bull power > 0 + above daily EMA + volume surge
            if (bullish_align[i] and bull_power[i] > 0 and 
                close[i] > ema1d and vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + bear power < 0 + below daily EMA + volume surge
            elif (bearish_align[i] and bear_power[i] < 0 and 
                  close[i] < ema1d and vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: alignment breaks or power reverses
            if position == 1:
                if not bullish_align[i] or bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not bearish_align[i] or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_AlligatorElderRay_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 05:04
