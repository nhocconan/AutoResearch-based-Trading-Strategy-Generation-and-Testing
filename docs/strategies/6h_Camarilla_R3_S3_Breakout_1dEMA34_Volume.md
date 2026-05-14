# Strategy: 6h_Camarilla_R3_S3_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.034 | +20.2% | -4.8% | 194 | FAIL |
| ETHUSDT | 0.198 | +28.0% | -6.8% | 171 | PASS |
| SOLUSDT | 0.312 | +39.6% | -17.0% | 154 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.083 | +17.7% | -6.1% | 64 | PASS |
| SOLUSDT | 0.688 | +13.1% | -3.6% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Uses 1d EMA34 for trend filter and 1d Camarilla levels (R3/S3) for breakout entries
# Entry: Long when price breaks above R3 AND price > 1d EMA34 (uptrend) AND volume spike
#        Short when price breaks below S3 AND price < 1d EMA34 (downtrend) AND volume spike
# Exit: Price crosses 1d EMA34 (trend reversal) OR price reverts to R2/S2 (mean reversion)
# Works in both bull and bear markets by trading breakouts with 1d trend filter
# Target: 75-150 total trades over 4 years (19-38/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3, R2, S2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d_arr) / 3
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 4
    s3_1d = pivot_1d - range_1d * 1.1 / 4
    r2_1d = pivot_1d + range_1d * 1.1 / 6
    s2_1d = pivot_1d - range_1d * 1.1 / 6
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price below 1d EMA34 (trend change) OR price reverts to R2 (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] < r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price above 1d EMA34 (trend change) OR price reverts to S2 (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] > s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 07:47
