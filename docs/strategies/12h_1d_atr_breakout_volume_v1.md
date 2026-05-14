# Strategy: 12h_1d_atr_breakout_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.390 | +43.3% | -9.2% | 20 | PASS |
| ETHUSDT | -0.429 | -11.4% | -26.6% | 19 | FAIL |
| SOLUSDT | 0.803 | +140.8% | -22.9% | 14 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.039 | +6.1% | -9.6% | 8 | PASS |
| SOLUSDT | -0.313 | -2.2% | -22.4% | 5 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_1d_atr_breakout_volume_v1
# Hypothesis: Trade breakouts of daily ATR-based channels with volume confirmation on 12h timeframe.
# In bullish regime (price > 50-period SMA): long when price breaks above upper ATR channel.
# In bearish regime (price < 50-period SMA): short when price breaks below lower ATR channel.
# Uses volume filter (1.5x average) to confirm breakout strength.
# ATR multiplier of 1.5 provides reasonable channel width to avoid whipsaws.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Works in both bull and bear markets by adapting to prevailing trend via SMA filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_atr_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily ATR for channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-based channels (similar to Keltner)
    atr_mult = 1.5
    upper_channel = close_1d + atr_mult * atr
    lower_channel = close_1d - atr_mult * atr
    
    # Align channels to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Daily 50-period SMA for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below lower channel
            if close[i] < lower_channel_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper channel
            if close[i] > upper_channel_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume surge and bullish trend
            if (close[i] > upper_channel_aligned[i] and vol_surge and 
                close[i] > sma50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume surge and bearish trend
            elif (close[i] < lower_channel_aligned[i] and vol_surge and 
                  close[i] < sma50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 18:27
