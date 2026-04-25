#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d ADX Trend + Volume Spike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance; ADX > 25 from 1d confirms strong trend; volume spike validates breakout conviction. Works in bull/bear by only trading in direction of 1d ADX trend, avoiding counter-trend entries. Targets 12-37 trades/year on 6h timeframe to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52) on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for entry to avoid look-ahead
    
    # 1d ADX for trend strength
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).rolling(2).max() - pd.Series(df_1d['low']).rolling(2).min()
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, ATR
    tr_ema = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.maximum(tr_ema, 1e-10))
    minus_di = 100 * (minus_dm_smooth / np.maximum(tr_ema, 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 + 26 shift)
    start_idx = 52 + 26  # Senkou B needs 52 periods, then shifted 26
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        # Cloud is plotted 26 periods ahead, so we use values calculated 26 periods ago
        idx_cloud = i - 26
        if idx_cloud < 0:
            signals[i] = 0.0
            continue
            
        senkou_a_cloud = senkou_a[idx_cloud]
        senkou_b_cloud = senkou_b[idx_cloud]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_cloud, senkou_b_cloud)
        cloud_bottom = min(senkou_a_cloud, senkou_b_cloud)
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # TK Cross (Tenkan/Kijun crossover)
        tk_cross_bull = tenkan[i] > kijun[i]
        tk_cross_bear = tenkan[i] < kijun[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals
            # Long: price above cloud + TK bullish cross + strong trend + volume spike
            long_entry = price_above_cloud and tk_cross_bull and strong_trend and vol_spike
            # Short: price below cloud + TK bearish cross + strong trend + volume spike
            short_entry = price_below_cloud and tk_cross_bear and strong_trend and vol_spike
            
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
            # Exit: price falls below cloud OR TK cross turns bearish OR trend weakens
            if (not price_above_cloud) or (not tk_cross_bull) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud OR TK cross turns bullish OR trend weakens
            if (not price_below_cloud) or (not tk_cross_bear) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0