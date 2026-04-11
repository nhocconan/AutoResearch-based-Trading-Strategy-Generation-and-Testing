#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_trend_v2
# Strategy: 6h Ichimoku Cloud trend following with 1d cloud filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku Cloud provides clear trend direction and support/resistance.
# Using 1d cloud as higher timeframe filter ensures alignment with major trend.
# Volume confirmation reduces false breakouts. Works in bull (cloud support) and bear (cloud resistance).
# Target: 15-30 trades/year on 6f (60-120 over 4 years)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 1d Ichimoku Cloud calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2).shift(26)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # 6h Ichimoku for entry timing
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_sen_6h.iloc[i]) or np.isnan(kijun_sen_6h.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # 6h TK cross
        tk_cross = tenkan_sen_6h.iloc[i] - kijun_sen_6h.iloc[i]
        tk_cross_prev = tenkan_sen_6h.iloc[i-1] - kijun_sen_6h.iloc[i-1]
        
        # Bullish TK cross: Tenkan crosses above Kijun
        bullish_tk = (tk_cross > 0) and (tk_cross_prev <= 0)
        # Bearish TK cross: Tenkan crosses below Kijun
        bearish_tk = (tk_cross < 0) and (tk_cross_prev >= 0)
        
        # Entry logic: TK cross + price outside cloud + volume + 1d trend filter
        if (bullish_tk and close[i] > cloud_top and vol_confirm[i] and 
            close[i] > kijun_sen_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (bearish_tk and close[i] < cloud_bottom and vol_confirm[i] and 
              close[i] < kijun_sen_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite TK cross or price re-enters cloud
        elif position == 1 and (bearish_tk or close[i] < cloud_top):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_tk or close[i] > cloud_bottom):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals