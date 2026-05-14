# Strategy: 4h_Donchian20_12hEMA_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.152 | +27.8% | -16.4% | 75 | PASS |
| ETHUSDT | 0.036 | +18.5% | -15.7% | 75 | PASS |
| SOLUSDT | 1.008 | +203.9% | -27.5% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.871 | -4.0% | -9.9% | 27 | FAIL |
| ETHUSDT | 0.347 | +12.0% | -9.2% | 24 | PASS |
| SOLUSDT | -0.284 | -1.7% | -18.4% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# Designed to capture sustained moves aligned with intermediate trend while filtering low-momentum breakouts.
# Uses discrete position sizing (0.25) to minimize fee drift. Target: 75-200 trades over 4 years.
# Works in bull/bear markets by following 12h EMA direction and requiring volume confirmation.

name = "4h_Donchian20_12hEMA_VolumeSpike"
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
    
    # Get 12h data for Donchian channels and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from completed 12h bars
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # Using shift(1) to ensure we only use completed 12h bars
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid Donchian and volume EMA
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + above 12h EMA50 + volume spike
            if close[i] > donchian_upper_aligned[i] and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + below 12h EMA50 + volume spike
            elif close[i] < donchian_lower_aligned[i] and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower or loses 12h trend alignment
            if close[i] < donchian_lower_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper or loses 12h trend alignment
            if close[i] > donchian_upper_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-03 20:58
