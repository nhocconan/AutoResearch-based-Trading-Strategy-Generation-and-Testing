#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dTrendFilter_VolumeSpike
Hypothesis: Ichimoku Tenkan-Kijun cross with 1d trend filter (price > 1d EMA50) and volume confirmation on 6h timeframe. 
Ichimoku provides dynamic support/resistance and trend direction. The TK cross captures momentum shifts, while the 1d EMA50 filter ensures alignment with higher timeframe trend. 
Volume spike confirms institutional participation. Designed for low trade frequency (<30/year) to avoid fee drag while capturing strong trending moves in both bull and bear markets.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    
    # Align Ichimoku components (no extra delay needed as they are concurrent indicators)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # self-align for same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Volume confirmation: 2.0x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku (52), 1d EMA (50), volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        
        # Determine Kumo (cloud) top and bottom
        if senkou_a_val >= senkou_b_val:
            kumo_top = senkou_a_val
            kumo_bottom = senkou_b_val
            kumo_color = 'green'  # bullish cloud
        else:
            kumo_top = senkou_b_val
            kumo_bottom = senkou_a_val
            kumo_color = 'red'    # bearish cloud
        
        # Price relative to cloud
        price_above_kumo = close_val > kumo_top
        price_below_kumo = close_val < kumo_bottom
        price_in_kumo = (close_val >= kumo_bottom) and (close_val <= kumo_top)
        
        # TK cross signals
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        if position == 0:
            # Long: TK cross up + price above/below cloud (bullish alignment) + volume confirmation + 1d uptrend
            long_signal = tk_cross_up and price_above_kumo and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_50_1d_val)
            # Short: TK cross down + price below/above cloud (bearish alignment) + volume confirmation + 1d downtrend
            short_signal = tk_cross_down and price_below_kumo and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below cloud OR 1d trend reversal
            if tk_cross_down or price_below_kumo or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud OR 1d trend reversal
            if tk_cross_up or price_above_kumo or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0