# Strategy: 4h_Williams_Alligator_Elder_Ray_Trend_12h

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.282 | +6.2% | -10.3% | 162 | FAIL |
| ETHUSDT | 0.339 | +41.0% | -11.9% | 131 | PASS |
| SOLUSDT | 0.821 | +125.7% | -20.6% | 149 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.849 | +19.7% | -7.7% | 44 | PASS |
| SOLUSDT | 0.040 | +5.7% | -11.2% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Williams_Alligator_Elder_Ray_Trend_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== 12h Trend Filter (HTF) =====
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # ===== Williams Alligator (LTF) =====
    # Jaw: 13-period SMMA, offset 8
    # Teeth: 8-period SMMA, offset 5
    # Lips: 5-period SMMA, offset 3
    # SMMA = smoothed moving average (EMA with alpha=1/period)
    close_s = pd.Series(close)
    jaw = close_s.ewm(alpha=1/13, adjust=False).mean()
    jaw = jaw.shift(8)  # offset 8
    teeth = close_s.ewm(alpha=1/8, adjust=False).mean()
    teeth = teeth.shift(5)  # offset 5
    lips = close_s.ewm(alpha=1/5, adjust=False).mean()
    lips = lips.shift(3)  # offset 3
    
    jaw = jaw.fillna(0).values
    teeth = teeth.fillna(0).values
    lips = lips.fillna(0).values
    
    # ===== Elder Ray Index (LTF) =====
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) + Bull Power > 0 + Bear Power rising + 12h EMA50 uptrend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Alligator bullish alignment
                bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear power rising (less negative)
                close[i] > ema50_12h_aligned[i] and  # Price above 12h EMA50
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Lips < Teeth < Jaw) + Bear Power < 0 + Bull Power falling + 12h EMA50 downtrend + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and  # Alligator bearish alignment
                  bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull power falling (less positive)
                  close[i] < ema50_12h_aligned[i] and  # Price below 12h EMA50
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR price crosses below 12h EMA50
            if lips[i] < teeth[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR price crosses above 12h EMA50
            if lips[i] > teeth[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 04:16
