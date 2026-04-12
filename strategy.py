#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Cloud_Breakout
Hypothesis: Ichimoku Cloud from daily timeframe provides dynamic support/resistance.
Tenkan/Kijun cross on 6h with price above/below daily cloud filters false breakouts.
Works in bull (cloud acts as support) and bear (cloud acts as resistance) by using cloud color.
Low trade frequency via cloud filter reduces whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, senkouA, senkouB"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Senkou Span A: (Tenkan + Kijun)/2 plotted 26 periods ahead
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B: (52-period high + low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    return tenkan, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === DAILY ICHIMOKU FOR CLOUD AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h (wait for daily bar to close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    # Cloud color: green if senkouA > senkouB (bullish), red otherwise
    cloud_green = senkou_a_1d_aligned > senkou_b_1d_aligned
    
    # === 6H TENKAN/KIJUN CROSS ===
    # Calculate Tenkan and Kijun on 6h data
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: TK cross up + price above cloud + cloud bullish + volume
        long_signal = tk_cross_up[i] and (close[i] > cloud_top[i]) and cloud_green[i] and (vol_ratio[i] > 1.5)
        
        # Short: TK cross down + price below cloud + cloud bearish + volume
        short_signal = tk_cross_down[i] and (close[i] < cloud_bottom[i]) and (not cloud_green[i]) and (vol_ratio[i] > 1.5)
        
        # Exit when TK cross reverses or price enters cloud
        exit_long = (tk_cross_down[i] and position == 1) or (close[i] < cloud_top[i] and close[i] > cloud_bottom[i] and position == 1)
        exit_short = (tk_cross_up[i] and position == -1) or (close[i] < cloud_top[i] and close[i] > cloud_bottom[i] and position == -1)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals