# Strategy: 4h_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike_ATRTrailingStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.107 | +24.9% | -11.2% | 137 | PASS |
| ETHUSDT | 0.338 | +37.7% | -15.3% | 105 | PASS |
| SOLUSDT | 0.676 | +76.9% | -16.7% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.169 | -4.2% | -10.6% | 47 | FAIL |
| ETHUSDT | 0.044 | +6.0% | -14.8% | 43 | PASS |
| SOLUSDT | 0.487 | +13.7% | -11.6% | 35 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 and close > 1w EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 and close < 1w EMA50 (bearish trend)
- Volume must be > 2.0x 20-period average for high-conviction breakouts
- ATR-based trailing stop: exit when price moves 2.5x ATR from extreme since entry
- Uses 1w HTF for trend filter (more stable than daily) to reduce whipsaw and capture major trend
- Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
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
    
    # Calculate Camarilla pivot levels (based on previous bar's OHLC)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous bar's OHLC to calculate current bar's Camarilla levels
        c = (high[i-1] + low[i-1] + close[i-1]) / 3
        r = high[i-1] - low[i-1]
        camarilla_h3[i] = c + (r * 1.1 / 2)  # H3 level
        camarilla_l3[i] = c - (r * 1.1 / 2)  # L3 level
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
    start_idx = max(20, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume spike
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume spike
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
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

name = "4h_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike_ATRTrailingStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 02:43
