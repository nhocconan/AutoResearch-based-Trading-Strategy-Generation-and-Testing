# Strategy: 6H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.250 | +5.1% | -16.4% | 76 | FAIL |
| ETHUSDT | 0.333 | +43.6% | -13.1% | 73 | PASS |
| SOLUSDT | 0.963 | +180.7% | -30.2% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.319 | +11.3% | -11.3% | 25 | PASS |
| SOLUSDT | 0.007 | +4.5% | -17.4% | 22 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Target: 12-37 trades/year per symbol. Uses discrete position sizing (0.25) to minimize fee churn.
Works in both bull/bear via trend filter and volume confirmation to avoid false breakouts.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 12h EMA50 = uptrend, close < 12h EMA50 = downtrend
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 6h volume > 1.8x 20-period MA (balanced to avoid overtrading)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest_20[i-1]  # Break below previous period's low
        
        if position == 0:
            # Long: Break above Donchian high AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: reverse signal or volatility-based stop (using ATR-like measure)
            exit_signal = False
            if position == 1:
                # Exit long on breakdown below Donchian low or strong adverse move
                if breakout_down or close[i] < lowest_20[i-1]:
                    exit_signal = True
            elif position == -1:
                # Exit short on breakout above Donchian high or strong adverse move
                if breakout_up or close[i] > highest_20[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 14:38
