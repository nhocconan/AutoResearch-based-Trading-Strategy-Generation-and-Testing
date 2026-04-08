# Strategy: 6h_ichimoku_cloud_1d_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.253 | +4.7% | -16.4% | 55 | FAIL |
| ETHUSDT | -0.157 | +5.2% | -25.5% | 43 | FAIL |
| SOLUSDT | 0.594 | +95.6% | -39.8% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.255 | +10.1% | -13.2% | 13 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_v1"
timezone = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Calculate Chikou Span (Lagging Span): Close shifted back by 26 periods
    chikou_span = pd.Series(high_1d).shift(-displacement)  # Using high for alignment, will be adjusted
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    chikou_span_6h = align_htf_to_ltf(prices, df_1d, chikou_span.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback for Ichimoku
    start_idx = max(52 + displacement, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(chikou_span_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (future cloud)
        senkou_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        senkou_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Current price vs cloud
        price_above_cloud = close[i] > senkou_top
        price_below_cloud = close[i] < senkou_bottom
        
        # TK Cross
        tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_bearish = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        # Price vs Kijun-sen (additional confirmation)
        price_above_kijun = close[i] > kijun_sen_6h[i]
        price_below_kijun = close[i] < kijun_sen_6h[i]
        
        if position == 1:  # Long position
            # Exit: price below cloud OR TK cross bearish
            if price_below_cloud or tk_cross_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above cloud OR TK cross bullish
            if price_above_cloud or tk_cross_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above cloud + TK cross bullish + price above Kijun
            if price_above_cloud and tk_cross_bullish and price_above_kijun:
                position = 1
                signals[i] = 0.25
            # Short: price below cloud + TK cross bearish + price below Kijun
            elif price_below_cloud and tk_cross_bearish and price_below_kijun:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 03:03
