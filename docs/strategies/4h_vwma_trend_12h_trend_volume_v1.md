# Strategy: 4h_vwma_trend_12h_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.073 | +14.2% | -11.9% | 250 | FAIL |
| ETHUSDT | 0.203 | +32.0% | -16.5% | 242 | PASS |
| SOLUSDT | 0.997 | +191.2% | -29.8% | 236 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.397 | -2.5% | -12.4% | 91 | FAIL |
| SOLUSDT | 0.268 | +10.3% | -10.3% | 75 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Volume-Weighted Moving Average (VWMA) Trend with 12h Trend Filter and Volume Confirmation
Hypothesis: VWMA captures institutional price levels better than SMA/EMA. Price above/below VWMA indicates trend direction. Filtered by 12h VWMA trend to avoid counter-trend trades and volume confirmation to avoid false signals. Works in bull/bear by aligning with higher timeframe trend. Targets 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_vwma_trend_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h VWMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    vwma_num_12h = (typical_price_12h * df_12h['volume'].values)
    vwma_den_12h = df_12h['volume'].values
    vwma_50_12h_num = pd.Series(vwma_num_12h).rolling(window=50, min_periods=50).sum().values
    vwma_50_12h_den = pd.Series(vwma_den_12h).rolling(window=50, min_periods=50).sum().values
    vwma_50_12h = np.divide(vwma_50_12h_num, vwma_50_12h_den, out=np.full_like(vwma_50_12h_num, np.nan), where=vwma_50_12h_den!=0)
    vwma_50_12h_aligned = align_htf_to_ltf(prices, df_12h, vwma_50_12h)
    
    # 4h VWMA(50) for entry signal
    typical_price = (high + low + close) / 3
    vwma_num = (typical_price * volume)
    vwma_den = volume
    vwma_50_num = pd.Series(vwma_num).rolling(window=50, min_periods=50).sum().values
    vwma_50_den = pd.Series(vwma_den).rolling(window=50, min_periods=50).sum().values
    vwma_50 = np.divide(vwma_50_num, vwma_50_den, out=np.full_like(vwma_50_num, np.nan), where=vwma_50_den!=0)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(vwma_50_12h_aligned[i]) or np.isnan(vwma_50[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below VWMA OR trend turns bearish
            if (close[i] <= vwma_50[i] or 
                close[i] <= vwma_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above VWMA OR trend turns bullish
            if (close[i] >= vwma_50[i] or 
                close[i] >= vwma_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above VWMA, uptrend, volume
            if (close[i] > vwma_50[i] and 
                close[i] > vwma_50_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below VWMA, downtrend, volume
            elif (close[i] < vwma_50[i] and 
                  close[i] < vwma_50_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 01:23
