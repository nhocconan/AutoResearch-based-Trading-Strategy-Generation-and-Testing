# Strategy: 6h_Ichimoku_Cloud_TK_Cross_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.927 | +6.8% | -4.2% | 152 | FAIL |
| ETHUSDT | 0.043 | +22.6% | -4.0% | 152 | PASS |
| SOLUSDT | 0.267 | +30.9% | -7.1% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.162 | +7.3% | -4.5% | 61 | PASS |
| SOLUSDT | -1.513 | -2.8% | -6.9% | 47 | FAIL |

## Code
```python
#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
Ichimoku provides multiple confirmation signals: Tenkan/Kijun cross for momentum,
Kumo (cloud) for support/resistance and trend direction. Using daily trend filter
avoids counter-trend trades. Volume spikes confirm institutional interest.
This should work in both bull and bear regimes by adapting to the daily trend.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Determine daily trend: price above/below cloud + TK cross
    # Bullish: price above cloud AND Tenkan > Kijun
    # Bearish: price below cloud AND Tenkan < Kijun
    bullish_trend = (close_1d > np.maximum(senkou_a_1d, senkou_b_1d)) & (tenkan_1d > kijun_1d)
    bearish_trend = (close_1d < np.minimum(senkou_a_1d, senkou_b_1d)) & (tenkan_1d < kijun_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate 6h Ichimoku for entry signals
    tenkan_6h, kijun_6h, _, _, _ = calculate_ichimoku(high, low, close, 9, 26, 52)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate cloud boundaries
        top_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        bottom_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK bullish cross, price above cloud, bullish daily trend, volume spike
            if (tenkan_6h[i] > kijun_6h[i] and  # TK bullish cross
                close[i] > top_cloud and         # Price above cloud
                bullish_aligned[i] > 0.5 and     # Bullish daily trend
                volume[i] > 1.8 * vol_avg_20[i]): # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: TK bearish cross, price below cloud, bearish daily trend, volume spike
            elif (tenkan_6h[i] < kijun_6h[i] and   # TK bearish cross
                  close[i] < bottom_cloud and        # Price below cloud
                  bearish_aligned[i] > 0.5 and       # Bearish daily trend
                  volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: TK cross in opposite direction or price returns to cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: TK bearish cross OR price drops below cloud
                if (tenkan_6h[i] < kijun_6h[i] or 
                    close[i] < top_cloud):
                    exit_signal = True
            else:  # position == -1
                # Exit short: TK bullish cross OR price rises above cloud
                if (tenkan_6h[i] > kijun_6h[i] or 
                    close[i] > bottom_cloud):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 17:37
