# Strategy: 12h_Donchian20_Breakout_1dATRRatio_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.756 | -0.2% | -13.5% | 12 | FAIL |
| ETHUSDT | 0.112 | +25.1% | -10.7% | 11 | PASS |
| SOLUSDT | -0.147 | +8.8% | -29.7% | 17 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.131 | +7.4% | -6.2% | 5 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
- Uses Donchian channel (20-period high/low) from 12h for breakout signals
- 1d ATR(14) / ATR(50) ratio < 0.8 defines low volatility regime (chop filter)
- Volume confirmation (> 1.3x 20-period average) filters low-momentum breakouts
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading breakouts in low-volatility regimes
- ATR regime filter prevents whipsaws during high volatility periods
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
    
    # Calculate 12h Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channel: 20-period high and low
    period20_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, period20_low)
    
    # Calculate 1d ATR regime filter (ATR(14) / ATR(50) < 0.8 = low volatility)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR calculations
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    atr_50 = wilders_smoothing(tr, 50)
    
    # ATR ratio: ATR(14) / ATR(50) < 0.8 indicates low volatility regime
    atr_ratio = np.where(atr_50 != 0, atr_14 / atr_50, np.nan)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian, ATR ratio, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_donchian = close[i] > donchian_high_aligned[i]
        price_below_donchian = close[i] < donchian_low_aligned[i]
        
        # Low volatility regime filter
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, low volatility, volume spike
            long_signal = (price_above_donchian and 
                          low_volatility and
                          volume[i] > 1.3 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian low, low volatility, volume spike
            short_signal = (price_below_donchian and 
                           low_volatility and
                           volume[i] > 1.3 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or volatility expansion
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or volatility expands
                if (price_below_donchian or 
                    atr_ratio_aligned[i] > 1.2):  # Volatility expansion
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or volatility expands
                if (price_above_donchian or 
                    atr_ratio_aligned[i] > 1.2):  # Volatility expansion
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRRatio_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 17:08
