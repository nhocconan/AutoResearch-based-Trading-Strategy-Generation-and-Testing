# Strategy: 4h_Camarilla_R1S1_Breakout_12hTrend_Volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.089 | +13.8% | -15.3% | 242 | FAIL |
| ETHUSDT | 0.157 | +28.3% | -16.6% | 231 | PASS |
| SOLUSDT | 0.848 | +139.3% | -29.0% | 222 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.750 | +19.6% | -11.7% | 72 | PASS |
| SOLUSDT | 0.071 | +6.1% | -15.3% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 4)
    R2 = pivot + (range_hl * 1.1 / 2)
    S1 = pivot - (range_hl * 1.1 / 4)
    S2 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Trend filter: 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for volume MA and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = R1_aligned[i]
        r2 = R2_aligned[i]
        s1 = S1_aligned[i]
        s2 = S2_aligned[i]
        trend = ema50_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above trend
            if close[i] > r1 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below trend
            elif close[i] < s1 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (reversion to mean)
            if close[i] < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 (reversion to mean)
            if close[i] > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 05:18
