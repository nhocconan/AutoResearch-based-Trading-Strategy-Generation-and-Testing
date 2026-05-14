# Strategy: 4h_12hSupertrend_1dDonchian_VolumeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.360 | -1.2% | -18.1% | 49 | FAIL |
| ETHUSDT | 0.023 | +17.8% | -20.1% | 40 | PASS |
| SOLUSDT | 0.970 | +199.0% | -31.8% | 42 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.260 | +9.9% | -12.0% | 15 | PASS |
| SOLUSDT | -1.037 | -14.2% | -20.8% | 17 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 1d Donchian breakout for entry.
# Supertrend from 12h timeframe filters trades to align with higher timeframe trend.
# Donchian breakout from 1d provides entry signals with high probability of continuation.
# Volume confirmation (>1.3x 20-period average) reduces false breakouts.
# ATR-based stop loss manages risk.
# Designed to work in both bull and bear markets by using 12h trend filter to avoid counter-trend trades.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Supertrend on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation
    atr_period = 10
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    factor = 3.0
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    supertrend = np.zeros_like(close_12h)
    dir_ = np.ones_like(close_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    dir_[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir_[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if dir_[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align 12h Supertrend to 4h timeframe
    supertrend_12h = supertrend
    dir_12h = dir_
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    dir_aligned = align_htf_to_ltf(prices, df_12h, dir_12h.astype(float))
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(dir_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above 1d Donchian high or below 1d Donchian low
            # Only trade in direction of 12h Supertrend (trend filter)
            
            # Long: price breaks above 1d Donchian high AND 12h Supertrend uptrend
            if (close[i] > donchian_high_aligned[i] and 
                dir_aligned[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1d Donchian low AND 12h Supertrend downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  dir_aligned[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1d Donchian low or 12h Supertrend turns down
            if (close[i] <= donchian_low_aligned[i] or 
                dir_aligned[i] == -1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1d Donchian high or 12h Supertrend turns up
            if (close[i] >= donchian_high_aligned[i] or 
                dir_aligned[i] == 1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hSupertrend_1dDonchian_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 07:41
