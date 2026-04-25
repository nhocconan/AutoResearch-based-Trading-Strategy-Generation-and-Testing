#!/usr/bin/env python3
"""
6h Ichimoku Cloud TK Cross + 1d ADX Trend Filter + Volume Spike
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) captures momentum shifts, 
filtered by 1d ADX > 25 for trending markets and volume confirmation. 
Works in bull markets (long on bullish TK cross above cloud) and bear markets 
(short on bearish TK cross below cloud) by using 1d ADX as regime filter. 
Designed for 6h timeframe with 50-150 total trades over 4 years to minimize 
fee drag while capturing strong momentum moves.
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
    
    # Get daily data for Ichimoku and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Ichimoku Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
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
    
    # Calculate ADX on daily data for trend strength filter
    period_adx = 14
    # Calculate True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period_adx, min_periods=period_adx).mean()
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the DM values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period_adx, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period_adx, adjust=False).mean() / atr)
    
    # Calculate ADX
    dx = (np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di))) * 100
    adx = pd.Series(dx).ewm(alpha=1/period_adx, adjust=False).mean().values
    
    # Align Ichimoku and ADX to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52 periods) and volume MA
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Trend filter: ADX > 25 for strong trend
        strong_trend = adx_val > 25
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # TK cross signals
        tk_cross_bullish = tenkan_val > kijun_val
        tk_cross_bearish = tenkan_val < kijun_val
        
        if position == 0:
            # Look for entry signals
            # Long: bullish TK cross above cloud in strong trend with volume
            long_signal = tk_cross_bullish and (curr_close > cloud_top) and strong_trend and volume_confirm
            # Short: bearish TK cross below cloud in strong trend with volume
            short_signal = tk_cross_bearish and (curr_close < cloud_bottom) and strong_trend and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: bearish TK cross OR price drops below cloud bottom
            if tk_cross_bearish or curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud top
            if tk_cross_bullish or curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_ADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0