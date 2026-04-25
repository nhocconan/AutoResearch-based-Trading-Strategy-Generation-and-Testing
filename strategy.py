#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_WeeklyTrend
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter and weekly trend alignment.
Targets 12-37 trades/year by requiring: 1) TK cross signals, 2) price above/below 1d cloud (trend filter), 3) alignment with weekly EMA50.
Ichimoku provides objective trend/momentum signals; daily cloud filters false 6h signals; weekly trend ensures positioning with higher timeframe bias.
Works in bull/bear via cloud acting as dynamic support/resistance and weekly trend preventing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Ichimoku components and cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not needed for signals)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Weekly data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d Ichimoku (52) and weekly EMA50 (50)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = min(span_a_aligned[i], span_b_aligned[i])
        
        # Trend filter: price relative to cloud and weekly EMA50
        above_cloud = curr_close > cloud_top
        below_cloud = curr_close < cloud_bottom
        in_cloud = (curr_close >= cloud_bottom) and (curr_close <= cloud_top)
        weekly_uptrend = curr_close > ema_50_1w_aligned[i]
        weekly_downtrend = curr_close < ema_50_1w_aligned[i]
        
        # TK Cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        # Avoid whipsaw: require cross to be sustained (previous bar also crossed)
        prev_tk_up = tenkan_aligned[i-1] > kijun_aligned[i-1] if i > 0 else False
        prev_tk_down = tenkan_aligned[i-1] < kijun_aligned[i-1] if i > 0 else False
        tk_cross_up_confirmed = tk_cross_up and prev_tk_up
        tk_cross_down_confirmed = tk_cross_down and prev_tk_down
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long: TK cross up, price above cloud, weekly uptrend, volume confirmation
            long_signal = tk_cross_up_confirmed and above_cloud and weekly_uptrend and volume_confirm[i]
            # Short: TK cross down, price below cloud, weekly downtrend, volume confirmation
            short_signal = tk_cross_down_confirmed and below_cloud and weekly_downtrend and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if TK cross down OR price breaks below cloud bottom OR weekly trend changes
            if tk_cross_down_confirmed or curr_close < cloud_bottom or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if TK cross up OR price breaks above cloud top OR weekly trend changes
            if tk_cross_up_confirmed or curr_close > cloud_top or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_WeeklyTrend"
timeframe = "6h"
leverage = 1.0