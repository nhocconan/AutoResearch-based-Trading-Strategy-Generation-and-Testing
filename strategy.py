#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1dRegime_v1
Hypothesis: Ichimoku cloud (TK cross + price/cloud relationship) with 1d ADX regime filter on 6h timeframe. 
Only trade when 1d ADX > 25 (trending market) to avoid whipsaws in ranging conditions. 
TK cross provides timely entries, cloud acts as dynamic support/resistance. 
Designed for 12-37 trades/year by requiring strong trend alignment and clear Ichimoku signals.
Works in both bull and bear markets by only trading with the higher timeframe trend regime.
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
    
    # Get 1d data for HTF regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) for regime filter - only trade in trending markets
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and ATR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Ichimoku components on 6h (primary timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_9 = 9
    high_9 = pd.Series(high).rolling(window=period_9, min_periods=period_9).max().values
    low_9 = pd.Series(low).rolling(window=period_9, min_periods=period_9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_26 = 26
    high_26 = pd.Series(high).rolling(window=period_26, min_periods=period_26).max().values
    low_26 = pd.Series(low).rolling(window=period_26, min_periods=period_26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_52 = 52
    high_52 = pd.Series(high).rolling(window=period_52, min_periods=period_52).max().values
    low_52 = pd.Series(low).rolling(window=period_52, min_periods=period_52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for entry as it's lagging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) and 1d ADX (14+14+14=42 for smoothing)
    start_idx = max(52, 42)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        adx_val = adx_1d_aligned[i]
        close_val = close[i]
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        trending_market = adx_val > 25
        
        # Ichimoku signals
        # Bullish: price above cloud AND Tenkan crosses above Kijun
        # Bearish: price below cloud AND Tenkan crosses below Kijun
        # Cloud top/bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # TK cross (need previous values)
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        tk_cross_bull = tenkan_val > kijun_val and tenkan_prev <= kijun_prev
        tk_cross_bear = tenkan_val < kijun_val and tenkan_prev >= kijun_prev
        
        if position == 0:
            # Long: price above cloud + TK cross bull + trending market
            long_signal = price_above_cloud and tk_cross_bull and trending_market
            
            # Short: price below cloud + TK cross bear + trending market
            short_signal = price_below_cloud and tk_cross_bear and trending_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud base OR Tenkan crosses below Kijun
            if close_val < cloud_bottom or (tenkan_val < kijun_val and tenkan_prev >= kijun_prev):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud top OR Tenkan crosses above Kijun
            if close_val > cloud_top or (tenkan_val > kijun_val and tenkan_prev <= kijun_prev):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_1dRegime_v1"
timeframe = "6h"
leverage = 1.0