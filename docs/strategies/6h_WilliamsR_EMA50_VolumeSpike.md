# Strategy: 6h_WilliamsR_EMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.241 | +30.6% | -8.5% | 69 | PASS |
| ETHUSDT | 0.250 | +31.7% | -11.6% | 47 | PASS |
| SOLUSDT | 0.724 | +84.3% | -12.7% | 39 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.343 | +3.0% | -4.6% | 21 | FAIL |
| ETHUSDT | 0.393 | +11.0% | -7.1% | 21 | PASS |
| SOLUSDT | -0.873 | -5.8% | -13.3% | 18 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day EMA50 filter and volume spike confirmation.
# Long when: Williams %R crosses above -20 (oversold bounce), price > EMA50(1d), volume > 2x 20-period average
# Short when: Williams %R crosses below -80 (overbought rejection), price < EMA50(1d), volume > 2x 20-period average
# Exit when Williams %R returns to opposite extreme (%R < -80 for longs, %R > -20 for shorts)
# Williams %R captures momentum reversals, EMA50 filters trend direction, volume spike confirms conviction.
# Target: 20-40 trades/year per symbol. Works in bull (buy oversold dips) and bear (sell overbought rallies).
name = "6h_WilliamsR_EMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema50 = ema50_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 from below, price > EMA50, volume spike
            if (wr > -20 and williams_r[i-1] <= -20 and 
                price > ema50 and vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 from above, price < EMA50, volume spike
            elif (wr < -80 and williams_r[i-1] >= -80 and 
                  price < ema50 and vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns below -80 (overbought territory)
            if wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns above -20 (oversold territory)
            if wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 01:00
