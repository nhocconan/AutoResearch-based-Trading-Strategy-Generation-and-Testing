#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Weekly Trend Filter and Volume Spike
Hypothesis: Ichimoku TK cross + price above/below cloud captures trend momentum.
Weekly EMA34 filter ensures alignment with higher timeframe trend (works in bull/bear).
Volume spike confirms institutional participation. Targets 50-150 trades over 4 years.
Uses 6h timeframe with 1w/1d HTF for trend and cloud calculation.
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
    
    # Get 1d data for Ichimoku calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h (need to shift Senkou spans back by 26 periods for lookback)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Get 1w EMA34 for weekly trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 24-period volume MA for 6h volume confirmation (4 bars = 1 day)
    vol_ma_24_6h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24_6h[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku and volume MA
    start_idx = max(52 + 26, 24)  # 52 for Senkou B, 26 shift, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_24_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ma_6h = vol_ma_24_6h[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross: Tenkan crosses Kijun
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Volume confirmation: current 6h volume > 2.0 * 24-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_6h
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross up AND price above cloud AND price > weekly EMA (uptrend) AND volume confirmation
            long_entry = (tk_cross_up and price_above_cloud and 
                         curr_close > ema_trend and volume_confirm)
            # Short: TK cross down AND price below cloud AND price < weekly EMA (downtrend) AND volume confirmation
            short_entry = (tk_cross_down and price_below_cloud and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: TK cross down OR price falls below cloud OR price falls below weekly EMA
            if (tk_cross_down or curr_close < cloud_bottom or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: TK cross up OR price rises above cloud OR price rises above weekly EMA
            if (tk_cross_up or curr_close > cloud_top or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_WeeklyEMA34_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0