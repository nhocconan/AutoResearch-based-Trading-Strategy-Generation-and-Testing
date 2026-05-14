# Strategy: 4h_Camarilla_R4S4_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.359 | +39.2% | -11.0% | 158 | PASS |
| ETHUSDT | 0.172 | +29.1% | -12.5% | 149 | PASS |
| SOLUSDT | 0.875 | +130.0% | -20.2% | 125 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.717 | -1.8% | -8.5% | 58 | FAIL |
| ETHUSDT | 1.646 | +37.4% | -7.7% | 44 | PASS |
| SOLUSDT | 0.017 | +5.5% | -8.8% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above Camarilla R4 AND price > 1d EMA34 with volume spike.
# Short when price breaks below Camarilla S4 AND price < 1d EMA34 with volume spike.
# Uses 1d EMA34 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 15-25/year) to reduce fee drag.
# Works in both bull and bear markets by following the 1d trend direction.
name = "4h_Camarilla_R4S4_1dEMA34_Volume"
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
    
    # Load 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from previous 1d bar to avoid lookahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R4/S4)
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long condition: break above Camarilla R4, in uptrend with volume spike
            long_condition = (close[i] > R4_aligned[i]) and uptrend and vol_spike[i]
            # Short condition: break below Camarilla S4, in downtrend with volume spike
            short_condition = (close[i] < S4_aligned[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S4 or trend turns down
            if (close[i] < S4_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R4 or trend turns up
            if (close[i] > R4_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 22:20
