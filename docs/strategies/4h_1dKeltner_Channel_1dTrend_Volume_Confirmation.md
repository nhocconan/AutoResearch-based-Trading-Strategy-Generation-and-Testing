# Strategy: 4h_1dKeltner_Channel_1dTrend_Volume_Confirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.357 | +1.3% | -21.8% | 47 | FAIL |
| ETHUSDT | 0.027 | +19.5% | -15.2% | 44 | PASS |
| SOLUSDT | 1.090 | +204.2% | -25.9% | 44 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.177 | +8.3% | -11.4% | 19 | PASS |
| SOLUSDT | -0.889 | -10.5% | -21.6% | 17 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_1dKeltner_Channel_1dTrend_Volume_Confirmation
# Uses 1d Keltner Channels (EMA20 + ATR(10)*2) for dynamic support/resistance with 1d trend filter.
# Long when price breaks above upper Keltner channel in uptrend with volume confirmation.
# Short when price breaks below lower Keltner channel in downtrend with volume confirmation.
# Exit when price crosses back through the EMA20 middle band.
# Designed for 4h timeframe to capture institutional levels with trend alignment.

name = "4h_1dKeltner_Channel_1dTrend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Keltner Channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA20 for Keltner middle band
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily ATR(10) for Keltner channel width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Keltner Channels: EMA20 ± ATR(10)*2
    upper_keltner = ema_20_1d + (2.0 * atr_10_1d)
    lower_keltner = ema_20_1d - (2.0 * atr_10_1d)
    
    # Align Keltner Channels and EMA20 to 4h timeframe
    upper_keltner_4h = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_4h = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_4h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily volume filter (20-period MA) - calculated on 4h volume but using daily concept
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_keltner_4h[i]) or np.isnan(lower_keltner_4h[i]) or 
            np.isnan(ema_20_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above upper Keltner channel with uptrend and volume
            if close[i] > upper_keltner_4h[i] and close[i] > ema_20_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below lower Keltner channel with downtrend and volume
            elif close[i] < lower_keltner_4h[i] and close[i] < ema_20_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price crosses back below EMA20 middle band
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and close[i] < ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above EMA20 middle band
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and close[i] > ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 00:06
