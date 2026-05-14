# Strategy: 6h_PremiumDiscount_Equilibrium_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.179 | +27.6% | -9.0% | 232 | PASS |
| ETHUSDT | 0.409 | +41.1% | -7.2% | 211 | PASS |
| SOLUSDT | 0.858 | +105.2% | -15.8% | 177 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.162 | -3.7% | -7.8% | 88 | FAIL |
| ETHUSDT | 0.478 | +12.5% | -7.6% | 79 | PASS |
| SOLUSDT | 0.085 | +6.7% | -9.6% | 63 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PremiumDiscount_Equilibrium_1dTrend_Volume"
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
    
    # Get daily data for equilibrium and trend
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Daily equilibrium price (mean of high and low)
    eq_price = (df_d['high'].values + df_d['low'].values) / 2
    
    # Daily EMA(34) for trend filter
    close_d = pd.Series(df_d['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    eq_price_aligned = align_htf_to_ltf(prices, df_d, eq_price)
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_series = pd.Series(volume)
    vol_ma30 = vol_series.rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(eq_price_aligned[i]) or np.isnan(ema34_d_aligned[i]) or 
            np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma30[i]
        
        if position == 0:
            # Long: Price above equilibrium with volume and above daily EMA trend
            if close[i] > eq_price_aligned[i] and vol_ok and close[i] > ema34_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below equilibrium with volume and below daily EMA trend
            elif close[i] < eq_price_aligned[i] and vol_ok and close[i] < ema34_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below equilibrium
            if close[i] < eq_price_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above equilibrium
            if close[i] > eq_price_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 09:41
