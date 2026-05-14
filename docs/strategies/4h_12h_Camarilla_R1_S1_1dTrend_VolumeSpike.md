# Strategy: 4h_12h_Camarilla_R1_S1_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.489 | +39.8% | -6.8% | 276 | PASS |
| ETHUSDT | 0.047 | +22.2% | -7.1% | 257 | PASS |
| SOLUSDT | 0.160 | +28.4% | -21.0% | 226 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.474 | -5.3% | -7.7% | 107 | FAIL |
| ETHUSDT | 1.519 | +25.9% | -6.0% | 88 | PASS |
| SOLUSDT | 0.138 | +7.4% | -6.5% | 71 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R1_S1_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # Calculate 12h CAMARILLA pivot levels from previous 12h bar's OHLC
    prev_12h_high = df_12h['high'].shift(1).values
    prev_12h_low = df_12h['low'].shift(1).values
    prev_12h_close = df_12h['close'].shift(1).values
    
    # Camarilla formula for R1 and S1
    range_12h = prev_12h_high - prev_12h_low
    camarilla_mult = 1.1 / 12  # ~0.0916667
    r1_12h = prev_12h_close + range_12h * camarilla_mult * 1
    s1_12h = prev_12h_close - range_12h * camarilla_mult * 1
    
    # Align Camarilla levels to 12h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Calculate 12-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 12)  # Need 30 for 1d EMA and 12 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_30_1d_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_30_1d_aligned[i]
        r1_level = r1_12h_aligned[i]
        s1_level = s1_12h_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above R1 with volume AND price > 1d EMA30 (uptrend)
            if close[i] > r1_level and vol > 2.0 * vol_ma_val and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 with volume AND price < 1d EMA30 (downtrend)
            elif close[i] < s1_level and vol > 2.0 * vol_ma_val and close[i] < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below R1 OR trend reverses (price < 1d EMA30)
            if close[i] < r1_level or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above S1 OR trend reverses (price > 1d EMA30)
            if close[i] > s1_level or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 03:36
