# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.044 | +21.4% | -10.2% | 326 | PASS |
| ETHUSDT | 0.069 | +22.0% | -13.6% | 311 | PASS |
| SOLUSDT | 0.661 | +98.1% | -21.6% | 278 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.443 | +0.8% | -7.1% | 112 | FAIL |
| ETHUSDT | 0.909 | +22.9% | -9.2% | 94 | PASS |
| SOLUSDT | 1.069 | +26.9% | -8.6% | 100 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (S3/R3) breakout with 1d EMA34 trend filter and volume spike.
# Uses daily pivot levels for institutional support/resistance, EMA for trend direction,
# and volume spike for confirmation. Designed to work in both bull (pullbacks to S3) and bear
# (rejections at R3). Target: 20-50 trades/year to avoid fee drag.
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    # Using previous day's H, L, C (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot levels
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 4h timeframe (use previous day's values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data for pivot/EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above S3 + 1d uptrend + volume spike
            if (price > S3_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R3 + 1d downtrend + volume spike
            elif (price < R3_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below S3 or trend reverses
            if price < S3_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R3 or trend reverses
            if price > R3_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 02:01
