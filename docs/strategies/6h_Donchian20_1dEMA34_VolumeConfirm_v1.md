# Strategy: 6h_Donchian20_1dEMA34_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.133 | +26.0% | -9.1% | 101 | KEEP |
| ETHUSDT | 0.286 | +35.1% | -11.9% | 89 | KEEP |
| SOLUSDT | 0.879 | +121.8% | -21.4% | 76 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.790 | -1.2% | -6.4% | 37 | DISCARD |
| ETHUSDT | 0.659 | +15.9% | -6.8% | 30 | KEEP |
| SOLUSDT | -0.002 | +5.4% | -8.4% | 27 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Uses 6h timeframe (primary) and 1d HTF for trend alignment (proven pattern from DB)
- Donchian channel from previous 20 completed 6h bars (structure-based breakout)
- Long when price breaks above upper Donchian AND price > 1d EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below lower Donchian AND price < 1d EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the middle of the Donchian channel (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 6h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Donchian channel (20-period high/low)
    # Use rolling window on 6h data, then align to 6h timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need enough data for Donchian
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper = max(high_6h over 20 periods)
    # Donchian lower = min(low_6h over 20 periods)
    high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (previous 20-bar Donchian available at open)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, low_20)
    donchian_mid = (donchian_high_aligned + donchian_low_aligned) / 2.0  # Middle for exit
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34, Donchian(20), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend AND volume confirmation
            if close[i] > donchian_high_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 05:50
