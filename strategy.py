#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK cross and cloud filter from daily timeframe
# Ichimoku provides a comprehensive trend/momentum system: TK cross signals momentum shifts,
# cloud acts as dynamic support/resistance. Daily cloud filter ensures alignment with higher
# timeframe trend, reducing whipsaws. Works in both bull/bear by using cloud color (bullish/bearish)
# as trend filter. Target: 20-40 trades/year on 6h via strict TK cross + cloud confluence.
# Position sizing: 0.25 to limit drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data for Ichimoku (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bullish = senkou_span_a_6h[i] > senkou_span_b_6h[i]  # Green cloud
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) AND price above cloud (bullish bias)
            if tenkan_sen_6h[i] > kijun_sen_6h[i] and close[i] > cloud_top:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) AND price below cloud (bearish bias)
            elif tenkan_sen_6h[i] < kijun_sen_6h[i] and close[i] < cloud_bottom:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross reverses OR price enters cloud (weakens trend)
            if position == 1:
                # Exit long: TK cross bearish OR price below cloud top
                if tenkan_sen_6h[i] < kijun_sen_6h[i] or close[i] < cloud_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: TK cross bullish OR price above cloud bottom
                if tenkan_sen_6h[i] > kijun_sen_6h[i] or close[i] > cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_CloudFilter_Daily"
timeframe = "6h"
leverage = 1.0