#!/usr/bin/env python3
# 6h_ichimoku_weekly_trend_daily_volume_v1
# Hypothesis: 6h strategy using weekly Ichimoku Cloud for primary trend filter (1w),
# daily volume confirmation (1d > 1.5x 20-day average), and 6h TK cross for entry timing.
# Weekly cloud acts as strong trend filter (only trade in weekly cloud direction).
# Daily volume > 1.5x 20-day average confirms institutional participation.
# 6h TK cross provides precise entry/exit timing within the trend.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year.
# Uses weekly and daily HTF data called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_weekly_trend_daily_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for Ichimoku Cloud (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly Ichimoku parameters
    period9_high_w = pd.Series(high_w).rolling(window=9, min_periods=9).max().values
    period9_low_w = pd.Series(low_w).rolling(window=9, min_periods=9).min().values
    tenkan_sen_w = (period9_high_w + period9_low_w) / 2.0
    
    period26_high_w = pd.Series(high_w).rolling(window=26, min_periods=26).max().values
    period26_low_w = pd.Series(low_w).rolling(window=26, min_periods=26).min().values
    kijun_sen_w = (period26_high_w + period26_low_w) / 2.0
    
    senkou_a_w = ((tenkan_sen_w + kijun_sen_w) / 2.0)
    
    period52_high_w = pd.Series(high_w).rolling(window=52, min_periods=52).max().values
    period52_low_w = pd.Series(low_w).rolling(window=52, min_periods=52).min().values
    senkou_b_w = ((period52_high_w + period52_low_w) / 2.0)
    
    # Align weekly Ichimoku to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_w)
    
    # Daily HTF data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 6h TK cross for entry timing
    tenkan_6h = (pd.Series(close).rolling(window=9, min_periods=9).max().values +
                 pd.Series(close).rolling(window=9, min_periods=9).min().values) / 2.0
    kijun_6h = (pd.Series(close).rolling(window=26, min_periods=26).max().values +
                pd.Series(close).rolling(window=26, min_periods=26).min().values) / 2.0
    tk_cross = tenkan_6h - kijun_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(tk_cross[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly cloud boundaries
        cloud_top = max(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom = min(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        # Daily volume confirmation: current daily volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR TK cross turns bearish
            if close[i] < cloud_bottom or tk_cross[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if close[i] > cloud_top or tk_cross[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price above cloud AND TK cross bullish (Tenkan > Kijun)
                if close[i] > cloud_top and tk_cross[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below cloud AND TK cross bearish (Tenkan < Kijun)
                elif close[i] < cloud_bottom and tk_cross[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals