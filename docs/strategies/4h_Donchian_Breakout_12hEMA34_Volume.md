# Strategy: 4h_Donchian_Breakout_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.020 | +16.5% | -22.2% | 313 | FAIL |
| ETHUSDT | 0.244 | +36.0% | -15.7% | 326 | PASS |
| SOLUSDT | 0.796 | +150.1% | -28.5% | 312 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.158 | +7.9% | -12.0% | 122 | PASS |
| SOLUSDT | -0.028 | +3.5% | -16.9% | 96 | FAIL |

## Code
```python
#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
Trades breakouts above/below 20-period Donchian channels in the direction of the 12h EMA trend.
Uses volume spike confirmation to avoid false breakouts. Designed for low trade frequency
(20-50 trades/year) to minimize fee drag and work in both bull and bear markets by aligning
with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA for trend filter (34-period)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h Donchian channels (20-period) - calculated on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above Donchian high, above 12h EMA (uptrend)
            if close[i] > donchian_high_aligned[i] and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 12h EMA (downtrend)
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian low or closes below 12h EMA
                if close[i] < donchian_low_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Donchian high or closes above 12h EMA
                if close[i] > donchian_high_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 18:34
