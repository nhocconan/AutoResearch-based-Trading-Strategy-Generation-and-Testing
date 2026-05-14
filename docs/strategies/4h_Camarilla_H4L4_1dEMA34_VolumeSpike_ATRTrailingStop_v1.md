# Strategy: 4h_Camarilla_H4L4_1dEMA34_VolumeSpike_ATRTrailingStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.266 | +33.6% | -11.3% | 149 | PASS |
| ETHUSDT | 0.757 | +76.6% | -13.1% | 135 | PASS |
| SOLUSDT | 0.813 | +118.6% | -18.2% | 110 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.106 | +4.7% | -6.8% | 51 | FAIL |
| ETHUSDT | 1.263 | +30.5% | -7.1% | 48 | PASS |
| SOLUSDT | 0.015 | +5.3% | -8.3% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla H4 level and close > 1d EMA34 (bullish trend)
- Short when price breaks below Camarilla L4 level and close < 1d EMA34 (bearish trend)
- Volume must be > 2.0x 20-period average for high-conviction breakouts
- ATR(14) trailing stop: exit when price moves 2.5x ATR from extreme since entry
- Uses 1d HTF for trend filter (proven effective across multiple experiments)
- Target: 75-150 total trades over 4 years (19-37/year) to minimize fee drag
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
    
    # Calculate Camarilla levels (based on previous day's range)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day (1 bar of 4h = 6h, but we need 24h = 6 bars of 4h)
        if i >= 6:
            prev_high = np.max(high[i-6:i])
            prev_low = np.min(low[i-6:i])
            prev_close = close[i-1]
            
            pivot[i] = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            camarilla_h4[i] = pivot[i] + range_val * 1.1 / 2.0  # H4 level
            camarilla_l4[i] = pivot[i] - range_val * 1.1 / 2.0  # L4 level
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    start_idx = max(6, 34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(pivot[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H4, trend up (close > EMA34), volume spike
            if close[i] > camarilla_h4[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla L4, trend down (close < EMA34), volume spike
            elif close[i] < camarilla_l4[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
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

name = "4h_Camarilla_H4L4_1dEMA34_VolumeSpike_ATRTrailingStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 02:47
