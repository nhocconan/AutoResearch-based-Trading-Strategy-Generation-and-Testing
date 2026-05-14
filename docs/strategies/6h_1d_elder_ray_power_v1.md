# Strategy: 6h_1d_elder_ray_power_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.296 | -2.0% | -18.5% | 137 | FAIL |
| ETHUSDT | 0.294 | +41.9% | -15.1% | 122 | PASS |
| SOLUSDT | 0.864 | +181.4% | -38.6% | 118 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.059 | +5.5% | -12.0% | 48 | PASS |
| SOLUSDT | 0.118 | +6.7% | -16.5% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_power_v1"
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
    
    # Calculate daily components for Elder Ray
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA(13) as the trend reference
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Align to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Elder Ray signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 (bulls in control) + price above EMA13 + volume confirmation
        if bull_power > 0 and price_close > ema13_aligned[i] and vol_confirm:
            enter_long = True
        
        # Short: Bear Power < 0 (bears in control) + price below EMA13 + volume confirmation
        if bear_power < 0 and price_close < ema13_aligned[i] and vol_confirm:
            enter_short = True
        
        # Exit conditions: power signal reverses
        exit_long = bull_power <= 0
        exit_short = bear_power >= 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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

# Hypothesis: Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13) on daily timeframe
# with 6h execution. Works in bull markets (Bull Power > 0) and bear markets (Bear Power < 0) by
# measuring actual bull/bear strength relative to trend. Volume confirmation ensures participation.
# EMA13 filter prevents counter-trend trades. Position size 0.25 limits drawdown. Target: 50-150 trades.
```

## Last Updated
2026-04-11 05:23
