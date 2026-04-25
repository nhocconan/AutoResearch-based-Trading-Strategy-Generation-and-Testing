#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d ADX Trend Filter + Volume Spike
Hypothesis: Ichimoku TK cross on 6h with cloud filter from 1d captures momentum with defined risk; ADX>25 on 1d confirms strong trend to avoid whipsaws in ranging markets; volume spike confirms institutional participation. Works in bull/bear by following 1d trend direction via cloud color and ADX filter. Targets 12-37 trades/year via strict entry conditions.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components on 6h (conversion line, base line, leading spans)
    # Conversion line (Tenkan-sen): (9-period high + low)/2
    period_9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period_9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period_9_high + period_9_low) / 2
    
    # Base line (Kijun-sen): (26-period high + low)/2
    period_26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period_26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period_26_high + period_26_low) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period_52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period_52_high + period_52_low) / 2)
    
    # Align Ichimoku components to 6h (no additional shift needed as alignment handles completed bar)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # 1d ADX for trend strength filter
    # Calculate +DI, -DI, DX
    period_adx = 14
    up_move = pd.Series(df_1d['high']).diff().values
    down_move = -pd.Series(df_1d['low']).diff().values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = abs(pd.Series(df_1d['high']).diff() - pd.Series(df_1d['low']).diff())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    atr_1d = pd.Series(true_range).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values / atr_1d)
    
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d EMA50 for trend direction (bullish if price > EMA50)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(52, 20, 50)  # Ichimoku, volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku signals
        tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i]
        
        # Cloud color: green if Senkou A > Senkou B (bullish), red if Senkou A < Senkou B (bearish)
        cloud_bull = senkou_a_aligned[i] > senkou_b_aligned[i]
        cloud_bear = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = curr_close > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = curr_close < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Trend filters
        strong_trend = adx_1d_aligned[i] > 25
        bullish_bias = curr_close > ema_50_1d_aligned[i]
        bearish_bias = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: TK bullish cross + price above cloud + bullish cloud + strong trend + bullish bias + volume spike
            long_entry = (tk_cross_bull and price_above_cloud and cloud_bull and 
                         strong_trend and bullish_bias and vol_spike)
            # Short: TK bearish cross + price below cloud + bearish cloud + strong trend + bearish bias + volume spike
            short_entry = (tk_cross_bear and price_below_cloud and cloud_bear and 
                          strong_trend and bearish_bias and vol_spike)
            
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
            # Exit: TK bearish cross OR price falls below cloud OR loss of bullish bias
            if (tk_cross_bear or not price_above_cloud or curr_close < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: TK bullish cross OR price rises above cloud OR loss of bearish bias
            if (tk_cross_bull or not price_below_cloud or curr_close > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0