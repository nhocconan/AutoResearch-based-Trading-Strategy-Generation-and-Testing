# Strategy: 6h_wick_reversal_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.111 | +16.4% | -8.7% | 226 | FAIL |
| ETHUSDT | 0.036 | +21.7% | -8.8% | 206 | PASS |
| SOLUSDT | 0.635 | +75.0% | -16.1% | 183 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.737 | +16.4% | -6.1% | 74 | PASS |
| SOLUSDT | 0.235 | +9.0% | -9.9% | 75 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_wick_reversal_volume_v1
Hypothesis: On 6h timeframe, enter long when price closes above prior bar's high with strong bullish rejection (lower wick > 2x upper wick) and volume > 1.5x average, enter short when price closes below prior bar's low with strong bearish rejection (upper wick > 2x lower wick) and volume > 1.5x average. Uses 1d trend filter (price above/below 50-period EMA) to avoid counter-trend trades. Designed for 15-25 trades/year to minimize fee drag while capturing exhaustion moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_wick_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or
            np.isnan(high[i-1]) or np.isnan(low[i-1])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        # Wick analysis: lower wick = close - low, upper wick = high - close
        lower_wick = close[i] - low[i]
        upper_wick = high[i] - close[i]
        
        # Avoid division by zero
        if upper_wick == 0:
            upper_wick = 0.001
        if lower_wick == 0:
            lower_wick = 0.001
            
        if position == 1:  # Long position
            # Exit: price closes below prior bar's low (break of structure)
            if close[i] < low[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above prior bar's high (break of structure)
            if close[i] > high[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: bullish rejection (lower wick > 2x upper wick) + close > prior high + price > 1d EMA50
                if (lower_wick > (2 * upper_wick) and 
                    close[i] > high[i-1] and 
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: bearish rejection (upper wick > 2x lower wick) + close < prior low + price < 1d EMA50
                elif (upper_wick > (2 * lower_wick) and 
                      close[i] < low[i-1] and 
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 19:55
