# Strategy: 6h_ichimoku_cloud_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.100 | -12.4% | -18.0% | 27 | FAIL |
| ETHUSDT | -0.318 | +3.4% | -17.5% | 26 | FAIL |
| SOLUSDT | 1.149 | +166.2% | -18.5% | 23 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.229 | +9.1% | -10.7% | 9 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_ichimoku_cloud_trend_v1
# Hypothesis: Ichimoku Cloud (Tenkan-sen/Kijun-sen cross + Kumo twist) on 1d timeframe
# provides high-probability trend direction for 60-minute entries. The cloud acts as
# dynamic support/resistance and the TK cross signals momentum shifts. This combines
# trend-following with institutional-grade support/resistance, working in both bull
# and bear markets by aligning with the higher timeframe trend while using the
# 6-hour timeframe for precise entry timing.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = df_1d['close'].values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Kumo twist: Senkou A crossing Senkou B (trend change signal)
    # We use the previous bar's values to avoid look-ahead
    senkou_a_prev = np.roll(senkou_a_aligned, 1)
    senkou_b_prev = np.roll(senkou_b_aligned, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    kumo_twist_bull = (senkou_a_aligned > senkou_b_aligned) & (senkou_a_prev <= senkou_b_prev)
    kumo_twist_bear = (senkou_a_aligned < senkou_b_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # Start from sufficient lookback (max of all Ichimoku periods)
    start_idx = max(52, 26) + 1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below cloud bottom OR TK cross turns bearish
            if close[i] < cloud_bottom[i] or (tenkan_aligned[i] < kijun_aligned[i] and 
                                              tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above cloud top OR TK cross turns bullish
            if close[i] > cloud_top[i] or (tenkan_aligned[i] > kijun_aligned[i] and 
                                           tenkan_aligned[i-1] <= kijun_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Price above cloud AND bullish TK cross AND bullish Kumo twist
            if (close[i] > cloud_top[i] and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                kumo_twist_bull[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price below cloud AND bearish TK cross AND bearish Kumo twist
            elif (close[i] < cloud_bottom[i] and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  kumo_twist_bear[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 08:31
