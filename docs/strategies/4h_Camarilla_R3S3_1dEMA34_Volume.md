# Strategy: 4h_Camarilla_R3S3_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.611 | +41.2% | -4.7% | 260 | PASS |
| ETHUSDT | 0.028 | +21.8% | -9.5% | 247 | PASS |
| SOLUSDT | 0.319 | +38.7% | -17.6% | 212 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.906 | -6.1% | -8.5% | 103 | FAIL |
| ETHUSDT | 0.912 | +15.9% | -8.3% | 89 | PASS |
| SOLUSDT | 0.315 | +9.0% | -4.8% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and 4h volume spike.
# Long when price breaks above R3 AND price > EMA34(1d) AND 4h volume > 2x 20-period average.
# Short when price breaks below S3 AND price < EMA34(1d) AND 4h volume > 2x 20-period average.
# Exit when price crosses back below R3 (for long) or above S3 (for short).
# Camarilla levels from 1d provide institutional support/resistance. EMA34 filters trend direction.
# Volume spike confirms institutional participation. Target: 80-120 total trades over 4 years (20-30/year).

name = "4h_Camarilla_R3S3_1dEMA34_Volume"
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
    
    # 4h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    camarilla_r3 = typical_price + (range_1d * 1.1 / 4)
    camarilla_s3 = typical_price - (range_1d * 1.1 / 4)
    
    # EMA34 on 1d close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, price > EMA34, volume spike
            long_cond = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema_34_aligned[i]) and volume_filter[i]
            # Short conditions: break below S3, price < EMA34, volume spike
            short_cond = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema_34_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below R3
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above S3
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 03:24
