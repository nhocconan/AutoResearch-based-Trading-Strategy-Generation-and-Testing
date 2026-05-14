# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.213 | +32.2% | -16.7% | 119 | PASS |
| ETHUSDT | 0.050 | +19.0% | -17.1% | 120 | PASS |
| SOLUSDT | 1.126 | +259.1% | -29.8% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.854 | -5.5% | -9.0% | 51 | FAIL |
| ETHUSDT | 0.539 | +16.7% | -9.7% | 43 | PASS |
| SOLUSDT | 0.368 | +13.2% | -10.5% | 40 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 12h EMA50 AND volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 12h EMA50 AND volume > 1.5x MA20.
# Exit when price crosses the 12h EMA50 in opposite direction.
# Uses 12h HTF for trend to reduce noise and overtrading. Volume confirmation reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
# Camarilla levels provide structured support/resistance; 12h EMA50 filters for primary trend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
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
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous bar's OHLC)
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels
            R3 = prev_close + range_ * 1.1 / 2
            S3 = prev_close - range_ * 1.1 / 2
        else:
            R3 = np.nan
            S3 = np.nan
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 12h EMA50 (bullish trend) AND volume confirm
            if (not np.isnan(R3) and 
                close[i] > R3 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 12h EMA50 (bearish trend) AND volume confirm
            elif (not np.isnan(S3) and 
                  close[i] < S3 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 12h EMA50 (trend change)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 12h EMA50 (trend change)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-14 02:41
