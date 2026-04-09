#!/usr/bin/env python3
# 6h_ichimoku_trend_follow_v1
# Hypothesis: 6h Ichimoku trend following with 1d HTF cloud filter and volume confirmation.
# Enters long when Tenkan > Kijun, price above cloud, and bullish 1d trend (price > 1d Kumo top).
# Enters short when Tenkan < Kijun, price below cloud, and bearish 1d trend (price < 1d Kumo bottom).
# Uses volume confirmation (>1.2x 20-period average) to avoid false breakouts.
# Designed for low turnover (target: 12-37 trades/year) to work in both bull and bear markets
# by following institutional trend structure with higher timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) top and bottom
    kumomax = np.maximum(senkou_a, senkou_b)
    kumomin = np.minimum(senkou_a, senkou_b)
    
    # 1d HTF Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (9-period) on 1d
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen (26-period) on 1d
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A on 1d
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B on 1d (52-period)
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # 1d Kumo (Cloud) top and bottom
    kumomax_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumomin_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Ichimoku components to 6h timeframe
    kumomax_1d_aligned = align_htf_to_ltf(prices, df_1d, kumomax_1d)
    kumomin_1d_aligned = align_htf_to_ltf(prices, df_1d, kumomin_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumomax[i]) or np.isnan(kumomin[i]) or
            np.isnan(kumomax_1d_aligned[i]) or np.isnan(kumomin_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirmed = volume[i] > 1.2 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price falls below Kumomax (cloud top) or Tenkan < Kijun (trend change)
            if close[i] < kumomax[i] or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Kumomin (cloud bottom) or Tenkan > Kijun (trend change)
            if close[i] > kumomin[i] or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and 1d HTF trend alignment
            if volume_confirmed:
                # Bullish condition: Tenkan > Kijun, price above cloud, and bullish 1d trend
                bullish_6h = tenkan[i] > kijun[i] and close[i] > kumomax[i]
                bullish_1d = close[i] > kumomax_1d_aligned[i]  # Price above 1d cloud top
                
                # Bearish condition: Tenkan < Kijun, price below cloud, and bearish 1d trend
                bearish_6h = tenkan[i] < kijun[i] and close[i] < kumomin[i]
                bearish_1d = close[i] < kumomin_1d_aligned[i]  # Price below 1d cloud bottom
                
                # Long: bullish 6h and 1d trend with volume confirmation
                if bullish_6h and bullish_1d:
                    position = 1
                    signals[i] = 0.25
                # Short: bearish 6h and 1d trend with volume confirmation
                elif bearish_6h and bearish_1d:
                    position = -1
                    signals[i] = -0.25
    
    return signals