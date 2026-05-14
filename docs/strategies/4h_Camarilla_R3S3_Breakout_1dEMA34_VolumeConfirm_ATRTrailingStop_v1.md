# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_ATRTrailingStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.020 | +18.5% | -10.9% | 213 | FAIL |
| ETHUSDT | 0.332 | +40.1% | -16.3% | 191 | PASS |
| SOLUSDT | 0.613 | +82.6% | -22.5% | 165 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.014 | +23.3% | -9.5% | 66 | PASS |
| SOLUSDT | 0.536 | +14.5% | -9.9% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 level AND close > 1d EMA34 (bullish trend)
- Short when price breaks below Camarilla S3 level AND close < 1d EMA34 (bearish trend)
- Volume must be > 1.8x 20-period average for confirmation (tighter than typical to reduce trades)
- ATR(14) trailing stop: exit when price moves 2.5x ATR from extreme since entry
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance that work in both ranging and trending markets
- EMA34 trend filter prevents counter-trend entries during strong moves
- Volume confirmation ensures breakout legitimacy
- Wider ATR stop (2.5x) reduces whipsaw in volatile conditions
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
    
    # Calculate Camarilla levels for current 4h bar using previous bar's range
    # Camarilla R3 = close_prev + 1.1 * (high_prev - low_prev) / 2
    # Camarilla S3 = close_prev - 1.1 * (high_prev - low_prev) / 2
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]  # first bar uses same values
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev) / 2
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 20-period average volume (tighter filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * vol_ma
    
    # ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, trend up (close > EMA34), volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla S3, trend down (close < EMA34), volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Long exit: price drops 2.5x ATR from highest high since entry
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price rises 2.5x ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_ATRTrailingStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 03:00
