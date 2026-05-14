# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.555 | +12.7% | -5.1% | 129 | FAIL |
| ETHUSDT | 0.177 | +25.1% | -2.4% | 107 | PASS |
| SOLUSDT | 0.207 | +26.9% | -4.9% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.313 | +7.7% | -2.0% | 33 | PASS |
| SOLUSDT | 0.726 | +10.6% | -1.7% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA trend filter and volume spike.
# Uses actual Camarilla pivot levels from daily high/low/close to identify institutional support/resistance.
# Long when price breaks above R3 with 1d uptrend and volume spike; short when breaks below S3 with 1d downtrend.
# Designed to work in both bull (follow 1d uptrend) and bear (follow 1d downtrend) markets.
# Target: 20-50 trades/year to avoid fee drag.
name = "4h_Camarilla_R3S3_Breakout_1dEMA_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as primary breakout levels
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period EMA (higher threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data for Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + 1d uptrend + volume spike
            if (price > r3_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + 1d downtrend + volume spike
            elif (price < s3_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R3 or trend reverses
            if price < r3_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S3 or trend reverses
            if price > s3_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 01:51
