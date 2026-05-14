# Strategy: 6h_Donchian20_Breakout_12hEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.094 | +12.8% | -16.3% | 73 | FAIL |
| ETHUSDT | 0.151 | +27.8% | -16.5% | 70 | PASS |
| SOLUSDT | 1.123 | +236.3% | -28.6% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.836 | +23.7% | -7.7% | 20 | PASS |
| SOLUSDT | -0.065 | +2.7% | -20.3% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-period average). Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed to capture medium-term trends in both bull and bear markets by combining price structure (Donchian), trend (12h EMA50), and volume strength. Targets 50-150 total trades over 4 years.

name = "6h_Donchian20_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Volume confirmation: > 1.8x 20-period average (balanced threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Donchian(20) - price channel breakout
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) - trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND close > 12h EMA50 (bullish trend) AND volume confirm
            if (close[i] > donchian_high[i-1] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND close < 12h EMA50 (bearish trend) AND volume confirm
            elif (close[i] < donchian_low[i-1] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 12h EMA50 (trend change) OR touches Donchian low (mean reversion)
            if close[i] < ema_50_12h_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 12h EMA50 (trend change) OR touches Donchian high (mean reversion)
            if close[i] > ema_50_12h_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-14 02:47
