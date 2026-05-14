# Strategy: 6h_ElderRay_Power_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.093 | +16.8% | -10.6% | 577 | FAIL |
| ETHUSDT | 0.284 | +34.4% | -9.9% | 514 | PASS |
| SOLUSDT | 0.653 | +80.6% | -18.6% | 461 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.181 | +8.0% | -9.4% | 190 | PASS |
| SOLUSDT | 0.319 | +10.0% | -9.8% | 174 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + Volume + 12h Trend Filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and rising + volume spike + 12h EMA50 uptrend
# Short when Bear Power > 0 and rising + volume spike + 12h EMA50 downtrend
# Works in bull (strong bull power continuations) and bear (strong bear power continuations)
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Smooth the power values to avoid noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Rising power: current > previous
    bull_rising = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bear_rising = bear_power_smooth > np.roll(bear_power_smooth, 1)
    # Handle first element
    bull_rising[0] = False
    bear_rising[0] = False
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Bull power > 0 and rising + volume + 12h uptrend
        if (bull_power_smooth[i] > 0 and bull_rising[i] and 
            volume[i] > vol_threshold[i] and close[i] > ema_12h_aligned[i]):
            signals[i] = 0.25
        
        # Short: Bear power > 0 and rising + volume + 12h downtrend
        elif (bear_power_smooth[i] > 0 and bear_rising[i] and 
              volume[i] > vol_threshold[i] and close[i] < ema_12h_aligned[i]):
            signals[i] = -0.25
        
        # Exit: power deteriorates or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (bull_power_smooth[i] <= 0 or not bull_rising[i] or close[i] <= ema_12h_aligned[i])) or
               (signals[i-1] == -0.25 and (bear_power_smooth[i] <= 0 or not bear_rising[i] or close[i] >= ema_12h_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_ElderRay_Power_Volume_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-15 06:49
