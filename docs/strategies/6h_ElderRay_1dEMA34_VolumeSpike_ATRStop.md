# Strategy: 6h_ElderRay_1dEMA34_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.288 | +36.0% | -12.7% | 274 | PASS |
| ETHUSDT | 0.471 | +53.6% | -13.3% | 257 | PASS |
| SOLUSDT | 1.262 | +255.2% | -25.9% | 222 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.070 | -6.6% | -9.3% | 97 | FAIL |
| ETHUSDT | 0.071 | +6.2% | -10.7% | 95 | PASS |
| SOLUSDT | 0.270 | +10.3% | -12.9% | 89 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
- Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
- Long: Bull Power > 0 (increasing) + price > 1d EMA34 + volume > 1.5x 20-period avg volume
- Short: Bear Power < 0 (decreasing) + price < 1d EMA34 + volume > 1.5x 20-period avg volume
- Exit: ATR trailing stop (2.5x ATR from extreme) OR Elder Power crosses zero
- Uses 1d EMA34 as trend filter to align with higher timeframe momentum
- Volume confirmation reduces false signals in ranging markets
- ATR trailing stop manages risk during strong trends
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA13 for Elder Ray (using 13-period EMA of close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13, 34)  # Need 20 for volume MA, 14 for ATR, 13 for EMA13, 34 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Elder Ray momentum conditions
        bull_increasing = bull_power[i] > bull_power[i-1]  # Bull Power rising
        bear_decreasing = bear_power[i] < bear_power[i-1]  # Bear Power falling (more negative)
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power increasing + price > 1d EMA34 + volume spike
            if bull_increasing and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Bear Power decreasing + price < 1d EMA34 + volume spike
            elif bear_decreasing and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Bull Power turns negative (momentum loss)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
            momentum_exit = bull_power[i] <= 0
            
            if trailing_stop_long or momentum_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from short extreme (trailing stop)
            # 2. Bear Power turns positive (momentum loss)
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            momentum_exit = bear_power[i] >= 0
            
            if trailing_stop_short or momentum_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 19:54
