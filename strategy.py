#!/usr/bin/env python3
"""
6h_ichimoku_1d_trend_volume_v1
Hypothesis: On 6h timeframe, enter long when Ichimoku Tenkan-sen crosses above Kijun-sen AND price is above Kumo cloud AND price is above 1d EMA200 (uptrend), with volume > 1.5x average. Enter short when Tenkan-sen crosses below Kijun-sen AND price is below Kumo cloud AND price is below 1d EMA200 (downtrend) with volume > 1.5x average. Uses 1d EMA200 for trend filter and Ichimoku for momentum and support/resistance. Target: 15-30 trades/year to minimize fee drift while capturing momentum in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo cloud boundaries (shifted forward 26 periods)
    # For signal at time t, we use Senkou Span A/B from t-26
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to shift
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a_shifted[i]) or np.isnan(senkou_span_b_shifted[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        # Kumo cloud top and bottom
        cloud_top = max(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        cloud_bottom = min(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        
        if position == 1:  # Long position
            # Exit: Tenkan-sen crosses below Kijun-sen OR price falls below cloud OR trend changes
            if (tenkan_sen[i] < kijun_sen[i] or 
                close[i] < cloud_bottom or 
                close[i] < ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan-sen crosses above Kijun-sen OR price rises above cloud OR trend changes
            if (tenkan_sen[i] > kijun_sen[i] or 
                close[i] > cloud_top or 
                close[i] > ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: Tenkan-sen crosses above Kijun-sen AND price above cloud AND uptrend
                if (tenkan_sen[i] > kijun_sen[i] and 
                    tenkan_sen[i-1] <= kijun_sen[i-1] and  # crossed above this bar
                    close[i] > cloud_top and 
                    close[i] > ema_200_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: Tenkan-sen crosses below Kijun-sen AND price below cloud AND downtrend
                elif (tenkan_sen[i] < kijun_sen[i] and 
                      tenkan_sen[i-1] >= kijun_sen[i-1] and  # crossed below this bar
                      close[i] < cloud_bottom and 
                      close[i] < ema_200_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals