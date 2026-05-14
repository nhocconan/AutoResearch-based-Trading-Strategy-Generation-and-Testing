# Strategy: 4h_Donchian20_1dEMA20_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.061 | +21.6% | -18.0% | 131 | PASS |
| ETHUSDT | 0.256 | +37.3% | -13.3% | 124 | PASS |
| SOLUSDT | 1.112 | +256.6% | -27.0% | 119 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.965 | -6.2% | -11.8% | 52 | FAIL |
| ETHUSDT | 0.498 | +15.5% | -10.1% | 45 | PASS |
| SOLUSDT | 0.369 | +13.1% | -13.8% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Get 4h data for Donchian breakout signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Volume confirmation: current volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_20_1d_aligned[i]
        downtrend = close[i] < ema_20_1d_aligned[i]
        
        # Breakout conditions: price breaks Donchian(20) channel
        long_breakout = close[i] > donch_high_20_aligned[i]
        short_breakout = close[i] < donch_low_20_aligned[i]
        
        # Entry conditions: require alignment of 1d trend and breakout with volume confirmation
        long_entry = long_breakout and uptrend and volume_confirm[i]
        short_entry = short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: reverse signal
        if position == 1:
            exit_condition = not uptrend
        elif position == -1:
            exit_condition = not downtrend
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "4h_Donchian20_1dEMA20_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 09:53
