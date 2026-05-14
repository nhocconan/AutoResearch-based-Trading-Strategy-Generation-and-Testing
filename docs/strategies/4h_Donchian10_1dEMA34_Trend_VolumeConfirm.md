# Strategy: 4h_Donchian10_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.311 | +37.3% | -12.0% | 191 | PASS |
| ETHUSDT | 0.001 | +17.0% | -16.2% | 187 | PASS |
| SOLUSDT | 0.778 | +126.0% | -29.0% | 186 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.726 | -2.4% | -8.3% | 71 | FAIL |
| ETHUSDT | 0.593 | +16.6% | -9.8% | 58 | PASS |
| SOLUSDT | 0.155 | +7.8% | -13.9% | 60 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) for volatility normalization
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channels (10-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (10-period)
    highest_high_4h = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Volume confirmation: current volume > 1.5x average volume (4h average)
    vol_ma_4h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_confirm = volume > vol_ma_4h * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 10, 10)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(highest_high_4h_aligned[i]) or
            np.isnan(lowest_low_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_4h_aligned[i]
        breakout_down = close[i] < lowest_low_4h_aligned[i]
        
        # Entry conditions: require trend + breakout + volume confirmation
        long_entry = uptrend and breakout_up and volume_confirm[i]
        short_entry = downtrend and breakout_down and volume_confirm[i]
        
        # Exit conditions: when trend reverses or opposite breakout
        if position == 1:
            exit_condition = not uptrend or breakout_down
        elif position == -1:
            exit_condition = not downtrend or breakout_up
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

name = "4h_Donchian10_1dEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 10:02
