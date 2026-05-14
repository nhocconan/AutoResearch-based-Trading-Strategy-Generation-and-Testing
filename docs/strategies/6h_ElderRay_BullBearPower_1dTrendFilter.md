# Strategy: 6h_ElderRay_BullBearPower_1dTrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.207 | +14.9% | -6.1% | 116 | FAIL |
| ETHUSDT | 0.214 | +29.9% | -10.0% | 115 | PASS |
| SOLUSDT | 0.675 | +72.3% | -24.9% | 99 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.083 | +6.7% | -6.0% | 27 | PASS |
| SOLUSDT | -0.727 | -1.7% | -9.1% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter
Hypothesis: Elder Ray (Bull/Bear Power) combined with 1d EMA50 trend filter and volume confirmation for 6h timeframe.
Long when Bull Power > 0, Bear Power < 0, price > 1d EMA50, and volume > 1.5x average.
Short when Bull Power < 0, Bear Power > 0, price < 1d EMA50, and volume > 1.5x average.
Uses ATR-based trailing stop (2.5x ATR from extreme) to manage risk.
Designed for low trade frequency (12-37/year) to avoid fee drag while capturing momentum in both bull and bear markets.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (50), EMA13 (13), volume MA (20), ATR (14)
    start_idx = max(50, 13, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_13_val = ema_13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, uptrend, volume confirmation
            long_signal = (bull_power_val > 0) and (bear_power_val < 0) and (close_val > ema_50_1d_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: Bull Power < 0, Bear Power > 0, downtrend, volume confirmation
            short_signal = (bull_power_val < 0) and (bear_power_val > 0) and (close_val < ema_50_1d_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (Elder Ray divergence)
            if (low_val < long_stop) or (bull_power_val < 0 and bear_power_val > 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (Elder Ray divergence)
            if (high_val > short_stop) or (bull_power_val > 0 and bear_power_val < 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:20
