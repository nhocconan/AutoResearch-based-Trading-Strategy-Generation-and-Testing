# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike_ATRTrailingStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.320 | +5.5% | -15.1% | 296 | FAIL |
| ETHUSDT | 0.187 | +29.8% | -11.8% | 270 | PASS |
| SOLUSDT | 0.500 | +66.1% | -22.6% | 219 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.031 | +22.3% | -7.7% | 91 | PASS |
| SOLUSDT | 0.640 | +16.4% | -11.5% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 and close > 12h EMA34 (bullish trend)
- Short when price breaks below Camarilla S3 and close < 12h EMA34 (bearish trend)
- Volume must be > 1.8x 20-period average for conviction
- ATR-based trailing stop: exit long when price drops 2.0x ATR from highest high since entry
- ATR-based trailing stop: exit short when price rises 2.0x ATR from lowest low since entry
- Works in bull/bear via trend filter and breakout structure with proper risk management
- Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
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
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous bar's OHLC to calculate current bar's Camarilla levels
        c = (high[i-1] + low[i-1] + close[i-1]) / 3
        r = high[i-1] - low[i-1]
        camarilla_r3[i] = c + (r * 1.1 / 4)  # R3 level
        camarilla_s3[i] = c - (r * 1.1 / 4)  # S3 level
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
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
    start_idx = max(20, 34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, trend up (close > EMA34), volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema_34_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla S3, trend down (close < EMA34), volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema_34_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Long exit: price drops 2.0x ATR from highest high since entry
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price rises 2.0x ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike_ATRTrailingStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 02:41
