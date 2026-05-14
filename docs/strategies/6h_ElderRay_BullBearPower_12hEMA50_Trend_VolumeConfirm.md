# Strategy: 6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.078 | +13.9% | -15.5% | 232 | FAIL |
| ETHUSDT | 0.151 | +27.8% | -17.7% | 236 | PASS |
| SOLUSDT | 1.298 | +292.1% | -28.4% | 195 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.147 | +7.7% | -12.5% | 79 | PASS |
| SOLUSDT | 0.000 | +4.4% | -14.2% | 75 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation
- Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
- Long: Bull Power > 0 AND price > 12h EMA50 AND volume > 1.5x 20-period average
- Short: Bear Power < 0 AND price < 12h EMA50 AND volume > 1.5x 20-period average
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
- Uses 12h EMA50 for trend alignment (avoids counter-trend whipsaws)
- Volume confirmation ensures institutional participation
- Works in both bull and bear markets by trading with the trend using institutional-grade measures
- Elder Ray measures bull/bear power behind price moves, effective in all regimes
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
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to LTF
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50)  # Elder Ray needs 13, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # Trend filter
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 + uptrend + volume confirmation
            if bull_strong and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + downtrend + volume confirmation
            elif bear_strong and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Opposite Elder Ray signal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power becomes negative
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bear Power becomes positive
                if bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 18:32
