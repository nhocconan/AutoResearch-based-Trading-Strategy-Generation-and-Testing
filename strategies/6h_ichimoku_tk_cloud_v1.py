#!/usr/bin/env python3
# 6h_ichimoku_tk_cloud_v1
# Hypothesis: 6h strategy using Ichimoku TK cross with cloud filter from 1d timeframe.
# Enters long when TK crosses above AND price above 1d cloud (bullish bias).
# Enters short when TK crosses below AND price below 1d cloud (bearish bias).
# Uses weekly ADX regime filter to avoid choppy markets (ADX < 20 = range, no trades).
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag.
# Works in bull/bear by using weekly trend filter and 1d cloud as dynamic support/resistance.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_tk_cloud_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d HTF data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For simplicity, we use the current cloud values (not shifted) for price position
    # In real Ichimoku, cloud is plotted 26 periods ahead, but for filtering we check
    # if price is above/below the current cloud boundaries
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align 1d Ichimoku to 6h timeframe (completed 1d candle only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # 1w HTF data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on weekly data
    # True Range
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w) - pd.Series(high_1w).shift(1)
    down_move = pd.Series(low_1w).shift(1) - pd.Series(low_1w)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_1w + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1w + 1e-10)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 6h indicators for TK cross
    # Tenkan-sen (6h): (9-period high + 9-period low)/2
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(tenkan_sen_6h[i]) or
            np.isnan(kijun_sen_6h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Weekly regime filter: only trade in trending markets (ADX >= 20)
        trending = adx_1w_aligned[i] >= 20
        
        if position == 1:  # Long position
            # Exit: TK cross bearish OR price falls below cloud bottom
            if (tenkan_sen_6h[i] < kijun_sen_6h[i]) or (close[i] < cloud_bottom_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross bullish OR price rises above cloud top
            if (tenkan_sen_6h[i] > kijun_sen_6h[i]) or (close[i] > cloud_top_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TK cross bullish AND price above cloud top (strong bullish)
            if (tenkan_sen_6h[i] > kijun_sen_6h[i]) and \
               (tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]) and \
               (close[i] > cloud_top_aligned[i]) and \
               trending:
                position = 1
                signals[i] = 0.25
            # Enter short: TK cross bearish AND price below cloud bottom (strong bearish)
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i]) and \
                 (tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]) and \
                 (close[i] < cloud_bottom_aligned[i]) and \
                 trending:
                position = -1
                signals[i] = -0.25
    
    return signals