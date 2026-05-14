# Strategy: 6h_Camarilla_R4S4_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.322 | +3.9% | -17.5% | 45 | FAIL |
| ETHUSDT | 0.020 | +19.5% | -13.2% | 37 | PASS |
| SOLUSDT | 0.723 | +101.2% | -20.2% | 34 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.246 | +9.3% | -7.5% | 12 | PASS |
| SOLUSDT | -1.001 | -12.9% | -23.9% | 13 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R4 level with 1d EMA34 uptrend and volume > 1.8x 20-period volume EMA
# Short when price breaks below Camarilla S4 level with 1d EMA34 downtrend and volume > 1.8x 20-period volume EMA
# Uses 1d HTF for trend to reduce whipsaw vs shorter HTF, targeting 12-30 trades/year on 6h.
# Volume spike filter (1.8x) is strict to avoid overtrading. Camarilla levels provide clear pivot structure.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "6h_Camarilla_R4S4_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high - low), S4 = close - 1.5*(high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    camarilla_r4 = close_1d_prev + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d_prev - 1.5 * (high_1d - low_1d)
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)  # Volume at least 1.8x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND 1d uptrend AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND 1d downtrend AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S4 OR 1d trend turns down
            if (close[i] < camarilla_s4_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R4 OR 1d trend turns up
            if (close[i] > camarilla_r4_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 16:44
