#!/usr/bin/env python3
"""
6h_Ichimoku_KumoTwist_1dTrend_WeeklyVolume_v1
Hypothesis: Ichimoku TK cross with 1d cloud filter and weekly volume confirmation captures trend continuation in both bull/bear markets. Uses 6h timeframe for balanced trade frequency (~15-25 trades/year) and discrete sizing (0.25) to minimize fee drag. Weekly volume spike ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (1d)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values + 
                      pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values) / 2)
    
    # Shift senkou spans forward by 26 periods (cloud)
    senkou_span_a = np.roll(senkou_span_a, 26)
    senkou_span_b = np.roll(senkou_span_b, 26)
    senkou_span_a[:26] = np.nan
    senkou_span_b[:26] = np.nan
    
    # Align Ichimoku to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load weekly data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly volume average
    vol_ma_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Ichimoku (52), EMA50 (50), weekly volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        trend_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_1w_aligned[i]
        vol_val = volume[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(senkou_a_val) or 
            np.isnan(senkou_b_val) or np.isnan(trend_val) or np.isnan(vol_ma_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Determine cloud relationship
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        in_cloud = (close_val >= cloud_bottom) and (close_val <= cloud_top)
        above_cloud = close_val > cloud_top
        below_cloud = close_val < cloud_bottom
        
        # TK cross conditions
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1]) if i > 0 else False
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1]) if i > 0 else False
        
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Volume confirmation: current weekly volume > 1.5 * 20-week average
        volume_confirm = vol_val > (vol_ma_val * 1.5)
        
        # Entry conditions
        long_condition = tk_cross_up and above_cloud and is_uptrend and volume_confirm
        short_condition = tk_cross_down and below_cloud and is_downtrend and volume_confirm
        
        # Exit conditions: opposite TK cross or trend reversal
        long_exit = (position == 1 and (tk_cross_down or not is_uptrend))
        short_exit = (position == -1 and (tk_cross_up or not is_downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_KumoTwist_1dTrend_WeeklyVolume_v1"
timeframe = "6h"
leverage = 1.0