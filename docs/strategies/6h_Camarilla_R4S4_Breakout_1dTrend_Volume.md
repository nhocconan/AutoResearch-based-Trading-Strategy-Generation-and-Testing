# Strategy: 6h_Camarilla_R4S4_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.270 | +34.9% | -11.0% | 90 | PASS |
| ETHUSDT | 0.299 | +39.9% | -15.0% | 85 | PASS |
| SOLUSDT | 0.657 | +103.6% | -31.9% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.821 | -3.9% | -8.5% | 32 | FAIL |
| ETHUSDT | 0.008 | +5.0% | -10.8% | 32 | PASS |
| SOLUSDT | -0.300 | -1.7% | -16.6% | 29 | FAIL |

## Code
```python
# 6h_Camarilla_R4S4_Breakout_1dTrend_Volume
# Hypothesis: Breakouts at R4/S4 levels (extreme sentiment) with 1d trend filter and volume confirmation.
# R4/S4 captures stronger momentum than R3/S3, reducing false breakouts. Works in bull (breakout continuation)
# and bear (mean reversion fails, trend filters losses). Target: 20-50 trades/year.
# Timeframe: 6h, HTF: 1d for trend and Camarilla calculation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R4S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R4, S4)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R4, S4)
    r4 = prev_close + 1.1 * (prev_high - prev_low) * 1  # R4 = C + 1.1*(H-L)*1
    s4 = prev_close - 1.1 * (prev_high - prev_low) * 1  # S4 = C - 1.1*(H-L)*1
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align all to 6h (primary timeframe)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r4_val = r4_6h[i]
        s4_val = s4_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R4 with volume and above trend
            if close[i] > r4_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S4 with volume and below trend
            elif close[i] < s4_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S4 (mean reversion to center)
            if close[i] < s4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R4 (mean reversion to center)
            if close[i] > r4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 06:50
