# Strategy: 6h_IchimokuTK_Cross_1dCloud_Filter_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.163 | +10.6% | -13.7% | 126 | DISCARD |
| ETHUSDT | 0.011 | +18.4% | -13.6% | 126 | KEEP |
| SOLUSDT | 0.827 | +135.9% | -18.4% | 108 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.354 | +11.7% | -9.4% | 40 | KEEP |
| SOLUSDT | 0.055 | +5.7% | -11.6% | 36 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK Cross + 1d Cloud Filter + Volume Spike
# Ichimoku TK Cross (Tenkan/Kijun) captures momentum shifts. 1d cloud ensures alignment with higher timeframe trend.
# Volume confirmation reduces false signals. Designed for 12-37 trades/year on 6h to minimize fee drag.
# Works in bull markets via bullish TK cross above 1d cloud and in bear markets via bearish TK cross below 1d cloud.

name = "6h_IchimokuTK_Cross_1dCloud_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for cloud filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).mean().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).mean().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).mean().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).mean().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).mean().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).mean().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Align 1d cloud to 6h timeframe (wait for completed 1d bar)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku components for TK Cross
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).mean().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).mean().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).mean().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).mean().values
    kijun = (high_26 + low_26) / 2
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long conditions: bullish TK cross (Tenkan > Kijun) AND price above cloud AND volume spike
            if (tenkan[i] > kijun[i] and 
                close[i] > cloud_top and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish TK cross (Tenkan < Kijun) AND price below cloud AND volume spike
            elif (tenkan[i] < kijun[i] and 
                  close[i] < cloud_bottom and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish TK cross OR price falls below cloud
            if tenkan[i] < kijun[i] or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud
            if tenkan[i] > kijun[i] or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 15:51
