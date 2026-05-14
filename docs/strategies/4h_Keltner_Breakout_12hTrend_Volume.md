# Strategy: 4h_Keltner_Breakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.086 | +15.1% | -11.6% | 218 | FAIL |
| ETHUSDT | 0.479 | +52.7% | -19.5% | 204 | PASS |
| SOLUSDT | 0.866 | +131.0% | -21.1% | 201 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.495 | +13.6% | -8.3% | 60 | PASS |
| SOLUSDT | 0.302 | +10.4% | -9.9% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Keltner_Breakout_12hTrend_Volume
# Hypothesis: Use Keltner Channel breakout on 4h with 12h EMA trend filter and volume confirmation.
# Enter long when price breaks above upper Keltner band (EMA20 + 2*ATR) with volume, short when breaks below lower band (EMA20 - 2*ATR) with volume, only in direction of 12h trend.
# Exit when price returns to EMA20 or trend reverses.
# Designed for low frequency (20-40 trades/year) by using 4h for signal and 12h for trend filter.

name = "4h_Keltner_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h data for Keltner Channel ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) on 4h
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate EMA(20) on 4h (middle line)
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    upper_keltner = ema_20_4h + 2 * atr_10
    lower_keltner = ema_20_4h - 2 * atr_10
    
    # Align Keltner components to 4h (wait for 4h bar to close)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_4h, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_4h, lower_keltner)
    
    # === 12h data for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(50) on 12h for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Keltner with volume, in uptrend
            if close[i] > upper_keltner_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner with volume, in downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to EMA20 or trend reverses
            if close[i] <= ema_20_4h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to EMA20 or trend reverses
            if close[i] >= ema_20_4h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 07:38
