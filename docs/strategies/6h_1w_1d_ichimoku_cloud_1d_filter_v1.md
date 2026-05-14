# Strategy: 6h_1w_1d_ichimoku_cloud_1d_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.261 | +8.5% | -13.0% | 105 | FAIL |
| ETHUSDT | 0.008 | +19.4% | -15.8% | 90 | PASS |
| SOLUSDT | 1.235 | +191.6% | -15.5% | 96 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.316 | +2.3% | -8.8% | 21 | FAIL |
| SOLUSDT | 0.525 | +12.1% | -6.5% | 19 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_1w_1d_ichimoku_cloud_1d_filter_v1
# Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d cloud filter for trend alignment.
# Long when Tenkan > Kijun and price above 1d Kumo cloud; short when Tenkan < Kijun and price below 1d cloud.
# Uses weekly trend filter to avoid counter-trend trades in strong weekly trends.
# Designed for 12-30 trades/year on 6h to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_ichimoku_cloud_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen: (9-period high + low) / 2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(9, n):
        period9_high[i] = np.max(high[i-9:i+1])
        period9_low[i] = np.min(low[i-9:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + low) / 2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(26, n):
        period26_high[i] = np.max(high[i-26:i+1])
        period26_low[i] = np.min(low[i-26:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (52-period high + low) / 2
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(52, n):
        period52_high[i] = np.max(high[i-52:i+1])
        period52_low[i] = np.min(low[i-52:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Get 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku cloud (Senkou Span A and B)
    # Tenkan-sen 1d: (9-period high + low) / 2
    period9_high_1d = np.full(len(df_1d), np.nan)
    period9_low_1d = np.full(len(df_1d), np.nan)
    for i in range(9, len(df_1d)):
        period9_high_1d[i] = np.max(high_1d[i-9:i+1])
        period9_low_1d[i] = np.min(low_1d[i-9:i+1])
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d: (26-period high + low) / 2
    period26_high_1d = np.full(len(df_1d), np.nan)
    period26_low_1d = np.full(len(df_1d), np.nan)
    for i in range(26, len(df_1d)):
        period26_high_1d[i] = np.max(high_1d[i-26:i+1])
        period26_low_1d[i] = np.min(low_1d[i-26:i+1])
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A 1d: (Tenkan + Kijun) / 2
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B 1d: (52-period high + low) / 2
    period52_high_1d = np.full(len(df_1d), np.nan)
    period52_low_1d = np.full(len(df_1d), np.nan)
    for i in range(52, len(df_1d)):
        period52_high_1d[i] = np.max(high_1d[i-52:i+1])
        period52_low_1d[i] = np.min(low_1d[i-52:i+1])
    senkou_span_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align 1d Ichimoku components to 6h
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA25 for trend filter
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(52, 26)  # Ensure Ichimoku is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or 
            np.isnan(ema25_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # 1w trend filter: price above/below EMA25
        uptrend_1w = close[i] > ema25_1w_aligned[i]
        downtrend_1w = close[i] < ema25_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun or price drops below cloud
            if tenkan[i] < kijun[i] or close[i] < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun or price rises above cloud
            if tenkan[i] > kijun[i] or close[i] > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Tenkan crosses above Kijun, price above cloud, and 1d uptrend
            if (tenkan[i] > kijun[i] and 
                close[i] > upper_cloud and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: Tenkan crosses below Kijun, price below cloud, and 1d downtrend
            elif (tenkan[i] < kijun[i] and 
                  close[i] < lower_cloud and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 17:20
