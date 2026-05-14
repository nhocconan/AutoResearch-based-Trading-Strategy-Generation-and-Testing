# Strategy: 6h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.154 | +16.6% | -6.8% | 294 | FAIL |
| ETHUSDT | 0.144 | +26.1% | -9.3% | 253 | PASS |
| SOLUSDT | 0.879 | +95.8% | -10.4% | 179 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.442 | +10.5% | -6.0% | 94 | PASS |
| SOLUSDT | 0.414 | +10.3% | -6.6% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R4 and close > 12h EMA50 with volume > 2.0x 20-bar average.
# Short when price breaks below S4 and close < 12h EMA50 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# R4/S4 levels act as strong breakout zones; combined with 12h trend filter and volume spike reduces false breakouts.
# Works in bull markets via breakouts and in bear markets via mean-reversion at extreme levels.

name = "6h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_VolumeSpike"
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
    
    lookback = 20  # for volume average
    
    # Calculate Camarilla levels (R4, S4) using previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each 1d bar: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R4 = close + (high - low) * 1.1 / 2
    # Camarilla S4 = close - (high - low) * 1.1 / 2
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4, close > 12h EMA50, volume spike
            if (high[i] > R4_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4, close < 12h EMA50, volume spike
            elif (low[i] < S4_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 OR volume drops below average
            if (low[i] < S4_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 OR volume drops below average
            if (high[i] > R4_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 21:49
