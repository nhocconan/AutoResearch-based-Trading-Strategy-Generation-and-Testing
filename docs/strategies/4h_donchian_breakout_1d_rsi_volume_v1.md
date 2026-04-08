# Strategy: 4h_donchian_breakout_1d_rsi_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.221 | +6.1% | -16.2% | 66 | FAIL |
| ETHUSDT | 0.045 | +19.6% | -22.3% | 59 | PASS |
| SOLUSDT | 0.395 | +60.3% | -36.7% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.153 | +7.8% | -12.2% | 27 | PASS |
| SOLUSDT | -0.496 | -5.9% | -17.4% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4-hour Donchian(20) breakout with 1-day RSI filter and volume confirmation
Hypothesis: Breakouts of Donchian(20) channels in the direction of the 1-day RSI trend,
confirmed by volume > 2x 20-period average, capture momentum with fewer whipsaws.
Designed for ~25-35 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day RSI(14) for trend filter
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI turns bearish (<50) OR price breaks below Donchian(10) low
            donchian_low = np.min(low[max(0, i-10):i+1])
            if (rsi_14_1d_aligned[i] < 50 or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI turns bullish (>50) OR price breaks above Donchian(10) high
            donchian_high = np.max(high[max(0, i-10):i+1])
            if (rsi_14_1d_aligned[i] > 50 or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian(20) channels - standard width for balance
            donchian_high = np.max(high[max(0, i-20):i])
            donchian_low = np.min(low[max(0, i-20):i])
            
            # Long: price breaks above Donchian(20) high + volume spike + RSI > 50
            if (close[i] > donchian_high and
                rsi_14_1d_aligned[i] > 50 and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(20) low + volume spike + RSI < 50
            elif (close[i] < donchian_low and
                  rsi_14_1d_aligned[i] < 50 and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 01:46
