# Strategy: 6h_1d_elder_ray_ma_cross_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.627 | -19.3% | -29.9% | 136 | FAIL |
| ETHUSDT | 0.195 | +31.7% | -29.4% | 109 | PASS |
| SOLUSDT | 0.328 | +50.9% | -43.6% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.573 | +18.2% | -10.8% | 43 | PASS |
| SOLUSDT | 0.002 | +3.6% | -17.0% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_ma_cross_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d Elder Ray and 13-period EMA
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 13-period EMA of close
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema_13
    
    # Shift by 1 to use only completed 1d bars
    bull_power = np.roll(bull_power, 1)
    bear_power = np.roll(bear_power, 1)
    ema_13 = np.roll(ema_13, 1)
    bull_power[0] = np.nan
    bear_power[0] = np.nan
    ema_13[0] = np.nan
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Calculate 6h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_13_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: Bull Power > 0 AND price > EMA50 (uptrend) with volume
        long_signal = volume_confirmed and (bull_power_aligned[i] > 0) and (price_close > ema_50[i])
        
        # Short conditions: Bear Power < 0 AND price < EMA50 (downtrend) with volume
        short_signal = volume_confirmed and (bear_power_aligned[i] < 0) and (price_close < ema_50[i])
        
        # Exit when Elder Power reverses
        exit_long = position == 1 and bull_power_aligned[i] <= 0
        exit_short = position == -1 and bear_power_aligned[i] >= 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Elder Ray + EMA50 trend filter on 6h with volume confirmation.
# Uses 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure
# bull/bear strength relative to the 13-day EMA. Enters long when Bull Power > 0
# and price above 6h EMA50 with volume confirmation (>1.5x average). Enters short
# when Bear Power < 0 and price below 6h EMA50 with volume confirmation.
# Exits when the respective power crosses zero. Works in both bull and bear markets
# by aligning with the dominant trend on higher timeframe. Target: 50-150 total trades
# over 4 years (12-37/year) to minimize fee drag on 6h timeframe. Elder Ray captures
# the underlying power behind price moves, reducing false signals. Volume confirmation
# ensures participation from market actors. EMA50 filter prevents counter-trend trades.
```

## Last Updated
2026-04-11 06:57
