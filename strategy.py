#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dADX25_Trend
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud color filter on 6h, combined with 1d ADX>25 trend filter, captures medium-term momentum with reduced whipsaw. Works in bull/bear via 1d ADX regime (only trade when trending). Target: 50-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index with min_periods"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift(1)).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Movement
    up = pd.Series(high).diff()
    down = -pd.Series(low).diff()
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_dm = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_dm = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    if len(high) < 52:
        return (np.full_like(high, np.nan),) * 4
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2).shift(26)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX>25 trend filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + volume MA (20)
    start_idx = max(52, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when 1d ADX > 25 (trending market)
        if adx_1d_aligned[i] <= 25:
            # In ranging market, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a[i]
        curr_senkou_b = senkou_b[i]
        
        # Cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Bullish cloud: Senkou A > Senkou B
        bullish_cloud = curr_senkou_a > curr_senkou_b
        # Bearish cloud: Senkou A < Senkou B
        bearish_cloud = curr_senkou_a < curr_senkou_b
        
        if position == 0:
            # Look for entry signals - require: TK cross + cloud alignment + volume spike
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_tk_cross = (curr_tenkan > curr_kijun) and (tenkan[i-1] <= kijun[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_tk_cross = (curr_tenkan < curr_kijun) and (tenkan[i-1] >= kijun[i-1])
            
            long_entry = bullish_tk_cross and bullish_cloud and volume_spike[i]
            short_entry = bearish_tk_cross and bearish_cloud and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below cloud or TK cross turns bearish
            bearish_tk_cross = (curr_tenkan < curr_kijun) and (tenkan[i-1] >= kijun[i-1])
            if curr_close < cloud_bottom or bearish_tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above cloud or TK cross turns bullish
            bullish_tk_cross = (curr_tenkan > curr_kijun) and (tenkan[i-1] <= kijun[i-1])
            if curr_close > cloud_top or bullish_tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dADX25_Trend"
timeframe = "6h"
leverage = 1.0