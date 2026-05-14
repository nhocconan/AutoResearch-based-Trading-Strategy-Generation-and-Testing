# Strategy: 6h_Keltner_Channel_Breakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.077 | +23.4% | -17.3% | 121 | PASS |
| ETHUSDT | 0.224 | +32.8% | -10.9% | 120 | PASS |
| SOLUSDT | 0.935 | +146.0% | -23.8% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.144 | -15.0% | -15.3% | 48 | FAIL |
| ETHUSDT | 0.285 | +10.0% | -7.8% | 41 | PASS |
| SOLUSDT | -0.403 | -2.3% | -17.4% | 43 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_Keltner_Channel_Breakout_12hTrend_Volume
# Hypothesis: Keltner Channel breakout on 6h with 12h EMA trend filter and volume confirmation.
# Keltner Channels adapt to volatility via ATR, providing dynamic support/resistance.
# The 12h EMA filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear markets
# by following the trend defined by higher timeframe.

name = "6h_Keltner_Channel_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 30-period EMA on 12h for trend direction
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # === Keltner Channel (20, 2.0) on 6h ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(20)
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Middle Line (EMA of close)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and Lower Bands
    upper_band = ema_20 + 2.0 * atr_20
    lower_band = ema_20 - 2.0 * atr_20
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_30_12h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 12h EMA
        trend_up = close[i] > ema_30_12h_aligned[i]
        trend_down = close[i] < ema_30_12h_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Keltner band with volume and higher timeframe uptrend
            if (close[i] > upper_band[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner band with volume and higher timeframe downtrend
            elif (close[i] < lower_band[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below middle line or higher timeframe trend changes
            if (close[i] < ema_20[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above middle line or higher timeframe trend changes
            if (close[i] > ema_20[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 07:08
