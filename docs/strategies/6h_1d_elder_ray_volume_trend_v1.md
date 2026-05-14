# Strategy: 6h_1d_elder_ray_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.135 | +17.5% | -10.0% | 168 | FAIL |
| ETHUSDT | 0.094 | +24.2% | -7.7% | 159 | PASS |
| SOLUSDT | 0.337 | +42.3% | -19.0% | 175 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.820 | +15.8% | -6.7% | 53 | PASS |
| SOLUSDT | 1.045 | +18.8% | -5.4% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Enter long when Bull Power > 0 and rising, Bear Power < 0, with 1d EMA(50) uptrend and volume expansion.
# Enter short when Bear Power < 0 and falling, Bull Power > 0, with 1d EMA(50) downtrend and volume expansion.
# Uses EMA(13) for Elder Ray and EMA(50) for 1d trend filter.
# Designed for 12-30 trades/year on 6h timeframe with focus on institutional participation.
# Volume filter ensures breakouts have conviction, reducing false signals.
# 1d trend filter prevents counter-trend trading in choppy markets.

name = "6h_1d_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = ema_13 - low   # EMA(13) - Low
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after EMA(13) period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray conditions with slope (1-bar change)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] > bear_power[i-1]  # Note: bear_power is negative, so rising = less negative
        
        # Entry conditions
        bullish_entry = (bull_power[i] > 0) and bull_power_rising and (bear_power[i] < 0) and vol_filter and is_uptrend
        bearish_entry = (bear_power[i] < 0) and bear_power_falling and (bull_power[i] > 0) and vol_filter and is_downtrend
        
        # Exit conditions: opposite Elder Ray signal
        exit_long = (bear_power[i] < 0) and bear_power_falling and (bull_power[i] > 0)
        exit_short = (bull_power[i] > 0) and bull_power_rising and (bear_power[i] < 0)
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 22:48
