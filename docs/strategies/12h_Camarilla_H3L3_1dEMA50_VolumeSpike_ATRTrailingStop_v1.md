# Strategy: 12h_Camarilla_H3L3_1dEMA50_VolumeSpike_ATRTrailingStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.179 | +29.0% | -9.3% | 53 | PASS |
| ETHUSDT | 0.072 | +22.3% | -10.6% | 48 | PASS |
| SOLUSDT | 0.754 | +122.7% | -28.0% | 45 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.489 | -0.7% | -9.7% | 21 | FAIL |
| ETHUSDT | 0.273 | +10.1% | -14.9% | 19 | PASS |
| SOLUSDT | -0.951 | -14.7% | -25.0% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 level and close > 1d EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 level and close < 1d EMA50 (bearish trend)
- Volume must be > 2.0x 20-period average for high-conviction breakouts
- ATR(14) trailing stop: exit when price moves 2.5x ATR from extreme since entry
- Uses 12h primary timeframe to target 50-150 trades over 4 years (12-37/year)
- Designed to work in both bull and bear markets via strong trend filter and breakout structure
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
    
    # Calculate Camarilla pivot levels (using previous bar's OHLC)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate today's levels (no look-ahead)
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        pivot[i] = (ph + pl + pc) / 3.0
        range_val = ph - pl
        
        camarilla_h3[i] = pc + range_val * 1.1 / 4.0
        camarilla_l3[i] = pc - range_val * 1.1 / 4.0
        camarilla_h4[i] = pc + range_val * 1.1 / 2.0
        camarilla_l4[i] = pc - range_val * 1.1 / 2.0
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
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
    start_idx = max(1, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume spike
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume spike
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
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
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price rises 2.5x ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeSpike_ATRTrailingStop_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-24 02:54
