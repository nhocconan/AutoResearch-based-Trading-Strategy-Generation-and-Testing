# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Novel

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.016 | +18.6% | -12.7% | 47 | DISCARD |
| ETHUSDT | 0.058 | +21.7% | -16.7% | 42 | KEEP |
| SOLUSDT | 1.269 | +205.8% | -15.3% | 31 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.929 | +21.8% | -8.3% | 13 | KEEP |
| SOLUSDT | -0.438 | -2.4% | -14.5% | 14 | DISCARD |

## Code
```python
#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Novel"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load 1d data ONCE for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # 2. 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 3. Calculate daily high/low/close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4. Camarilla levels: R3, S3
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.25
    s3 = close_1d - hl_range * 1.25
    
    # 5. Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6. Volume filter: 50-period EMA for higher threshold
    vol_ema50 = pd.Series(volume).ewm(span=50, min_periods=50, adjust=False).mean().values
    volume_ok = volume > vol_ema50 * 2.0  # Adjusted threshold for 6h
    
    # 7. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 1d EMA34 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 1d EMA34 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses
                if close[i] < s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses
                if close[i] > r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals
```

## Last Updated
2026-05-11 09:32
