# Strategy: 4h_Donchian_20_1dEMA34_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.231 | +31.1% | -9.8% | 170 | PASS |
| ETHUSDT | 0.166 | +28.5% | -12.5% | 167 | PASS |
| SOLUSDT | 0.702 | +97.9% | -20.4% | 171 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.821 | -1.5% | -8.6% | 63 | FAIL |
| ETHUSDT | 0.420 | +12.0% | -7.1% | 56 | PASS |
| SOLUSDT | 0.327 | +10.7% | -8.0% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 1d trend filter and volume confirmation.
# Uses Donchian(20) from the previous 4h bar to identify breakouts. Breakouts above upper band or below lower band
# trigger entries in the direction of the 1d EMA(34) trend. Volume confirmation (1.5x 20-period average)
# filters false breakouts. Designed for 4h timeframe with ~20-50 total trades per year to minimize fee drag.

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from previous 4h bar (to avoid look-ahead)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Shift by 1 to use previous bar's high/low
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_high_4h[0] = np.nan  # First value invalid
    prev_low_4h[0] = np.nan
    
    # Calculate rolling max/min of previous 20 periods
    donchian_high = pd.Series(prev_high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (they are constant for the bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions: breakout from Donchian bands in trend direction with volume
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        long_entry = long_breakout and uptrend and volume_confirm[i]
        short_entry = short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: price returns to Donchian midpoint or trend reversal
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        long_exit = close[i] < donchian_mid
        short_exit = close[i] > donchian_mid
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_20_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 09:29
