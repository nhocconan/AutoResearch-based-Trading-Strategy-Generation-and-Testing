# Strategy: 6h_WilliamsR_1dEMA34_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.654 | +6.9% | -12.3% | 66 | FAIL |
| ETHUSDT | 0.305 | +31.4% | -9.8% | 54 | PASS |
| SOLUSDT | 0.769 | +61.0% | -7.3% | 38 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.267 | +8.8% | -5.7% | 20 | PASS |
| SOLUSDT | -1.999 | -5.4% | -7.5% | 12 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions. In strong trends, it can remain
# extreme, but pullbacks to 50 level with trend alignment provide high-probability entries.
# Volume spike confirms institutional participation. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_WilliamsR_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll use the inverse: (Close - Lowest Low) / (Highest High - Lowest Low) * 100 - 100
    # So %R = 0 is overbought, %R = -100 is oversold
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    williams_r = ((highest_high - close) / hh_ll) * -100  # Standard Williams %R
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -50 (from oversold) + 1d uptrend + volume spike
            if williams_r[i] > -50 and williams_r[i-1] <= -50 and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -50 (from overbought) + 1d downtrend + volume spike
            elif williams_r[i] < -50 and williams_r[i-1] >= -50 and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 or trend reversal
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 or trend reversal
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 23:15
