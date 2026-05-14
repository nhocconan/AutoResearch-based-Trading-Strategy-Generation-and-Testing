# Strategy: 6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.086 | +15.3% | -15.5% | 215 | DISCARD |
| ETHUSDT | 0.119 | +25.6% | -12.6% | 199 | KEEP |
| SOLUSDT | 1.106 | +186.1% | -18.8% | 173 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.543 | +15.0% | -9.5% | 61 | KEEP |
| SOLUSDT | -0.050 | +4.0% | -12.6% | 59 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray with 1d EMA50 Trend Filter and Volume Confirmation
- Uses 6h Williams Alligator (Jaw=13, Teeth=8, Lips=5) to define trend alignment
- 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
- 1d EMA50 defines long-term trend filter: only trade in direction of daily trend
- Volume confirmation (> 1.5x 20-period average) filters weak signals
- Exit when Alligator lines reverse or Elder Power weakens
- Designed for 6h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in both bull and bear markets by combining trend following with volatility filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams Alligator
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).ewm(span=jaw_period, min_periods=jaw_period, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=teeth_period, min_periods=teeth_period, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=lips_period, min_periods=lips_period, adjust=False).mean().values
    
    # Calculate 6h Elder Ray (using EMA13 as reference)
    ema13 = jaw  # Jaw is EMA13
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray strength: positive bull power = buying pressure, negative bear power = selling pressure
        strong_bull = bull_power[i] > 0
        strong_bear = bear_power[i] > 0
        
        if position == 0:
            # Long: Alligator bullish AND Elder Ray bullish AND above 1d EMA50 AND volume spike
            if (alligator_bullish and strong_bull and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Elder Ray bearish AND below 1d EMA50 AND volume spike
            elif (alligator_bearish and strong_bear and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses OR Elder Power weakens OR price crosses 1d EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when Alligator turns bearish OR bull power fades OR below 1d EMA50
                if (not alligator_bullish or not strong_bull or close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator turns bullish OR bear power fades OR above 1d EMA50
                if (not alligator_bearish or not strong_bear or close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 16:30
