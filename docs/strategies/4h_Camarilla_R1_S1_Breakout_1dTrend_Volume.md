# Strategy: 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.268 | +32.2% | -6.5% | 276 | KEEP |
| ETHUSDT | 0.133 | +26.4% | -9.1% | 265 | KEEP |
| SOLUSDT | 0.574 | +71.9% | -17.2% | 240 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.669 | -8.5% | -9.5% | 108 | DISCARD |
| ETHUSDT | 1.074 | +22.1% | -8.8% | 91 | KEEP |
| SOLUSDT | 1.099 | +23.0% | -10.7% | 75 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla pivot and trend
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla pivot calculation
    daily_high = df_d['high'].values
    daily_low = df_d['low'].values
    daily_close = df_d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = pivot + (daily_high - daily_low) * 1.1 / 12
    s1 = pivot - (daily_high - daily_low) * 1.1 / 12
    
    # Align to 4h timeframe (Camarilla levels from previous day)
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    
    # Daily EMA(34) for trend filter
    close_d = pd.Series(daily_close)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above daily EMA trend
            if close[i] > r1_aligned[i] and vol_ok and close[i] > ema34_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below daily EMA trend
            elif close[i] < s1_aligned[i] and vol_ok and close[i] < ema34_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below S1 (reversion to mean)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above R1 (reversion to mean)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 02:15
