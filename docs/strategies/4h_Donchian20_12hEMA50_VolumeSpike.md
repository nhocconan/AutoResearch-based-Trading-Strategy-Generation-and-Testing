# Strategy: 4h_Donchian20_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.005 | +18.6% | -17.9% | 109 | FAIL |
| ETHUSDT | 0.516 | +58.7% | -12.1% | 100 | PASS |
| SOLUSDT | 0.975 | +175.0% | -30.0% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.205 | +8.8% | -9.8% | 39 | PASS |
| SOLUSDT | 0.262 | +10.0% | -16.0% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Donchian breakouts capture momentum in trending markets, EMA50 on 12h filters for trend direction,
# and volume spikes (>2x average) confirm institutional interest. Works in both bull and bear
# markets by allowing long/short entries. Target: 75-200 total trades over 4 years (19-50/year).
name = "4h_Donchian20_12hEMA50_VolumeSpike"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    # Highest high and lowest low over past 20 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for EMA50 and 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_12h = ema_50_12h_aligned[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > Donchian high AND price > 12h EMA50 (uptrend) AND volume > 2x average
            if close[i] > donchian_high and close[i] > ema_12h and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < Donchian low AND price < 12h EMA50 (downtrend) AND volume > 2x average
            elif close[i] < donchian_low and close[i] < ema_12h and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < Donchian low OR trend reverses (price < 12h EMA50)
            if close[i] < donchian_low or close[i] < ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > Donchian high OR trend reverses (price > 12h EMA50)
            if close[i] > donchian_high or close[i] > ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 02:44
