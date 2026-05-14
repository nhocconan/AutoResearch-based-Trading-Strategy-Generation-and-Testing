# Strategy: 6h_Donchian20_1dEMA34_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.283 | +31.5% | -7.6% | 105 | PASS |
| ETHUSDT | 0.440 | +42.0% | -8.5% | 94 | PASS |
| SOLUSDT | 0.564 | +66.2% | -20.4% | 82 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.481 | +2.2% | -4.0% | 40 | FAIL |
| ETHUSDT | 0.904 | +17.6% | -5.7% | 33 | PASS |
| SOLUSDT | 0.664 | +14.8% | -6.7% | 28 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR trailing stop.
- Long: price breaks above Donchian upper (20-period 6h high) + price > 1d EMA34 + volume > 2.0x 20-period avg volume
- Short: price breaks below Donchian lower (20-period 6h low) + price < 1d EMA34 + volume > 2.0x 20-period avg volume
- Exit: trailing stop (2.0x ATR from extreme) OR Donchian breakout in opposite direction
- Uses 1d EMA34 as trend filter for better regime adaptation vs shorter EMAs
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 2.0x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_upper = pd.Series(df_6h['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_6h['low']).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20, 34)  # Need 20 for Donchian, 14 for ATR, 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_upper_aligned[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower_aligned[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d EMA34 + volume spike
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian breakout down + price < 1d EMA34 + volume spike
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Donchian breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            breakout_down_exit = close[i] < donchian_lower_aligned[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Donchian breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            breakout_up_exit = close[i] > donchian_upper_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 19:48
