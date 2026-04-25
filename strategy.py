#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross + Volume Spike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK cross (Tenkan/Kijun) signals momentum shifts.
Long when price above cloud + TK cross bullish + volume spike. Short when price below cloud + TK cross bearish + volume spike.
Cloud from 1d timeframe provides higher-timeframe structure. Works in bull via trend continuation, bear via counter-trend cloud breaks.
Discrete sizing (0.25) controls drawdown and fee churn. Target: 12-30 trades/year on 6h.
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
    
    # Get 1d data for Ichimoku cloud (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_9 = np.full(len(high_1d), np.nan)
    min_low_9 = np.full(len(low_1d), np.nan)
    for i in range(period_tenkan-1, len(high_1d)):
        max_high_9[i] = np.max(high_1d[i-(period_tenkan-1):i+1])
        min_low_9[i] = np.min(low_1d[i-(period_tenkan-1):i+1])
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = np.full(len(high_1d), np.nan)
    min_low_26 = np.full(len(low_1d), np.nan)
    for i in range(period_kijun-1, len(high_1d)):
        max_high_26[i] = np.max(high_1d[i-(period_kijun-1):i+1])
        min_low_26[i] = np.min(low_1d[i-(period_kijun-1):i+1])
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = np.full(len(high_1d), np.nan)
    min_low_52 = np.full(len(low_1d), np.nan)
    for i in range(period_senkou_b-1, len(high_1d)):
        max_high_52[i] = np.max(high_1d[i-(period_senkou_b-1):i+1])
        min_low_52[i] = np.min(low_1d[i-(period_senkou_b-1):i+1])
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku, ATR, and volume MA to propagate
    start_idx = max(52, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Cloud boundaries (Senkou Span A/B form the cloud)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # TK cross signals
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price above cloud + TK cross bullish + volume spike
            long_condition = (curr_close > cloud_top) and tk_bullish and volume_spike
            # Short: price below cloud + TK cross bearish + volume spike
            short_condition = (curr_close < cloud_bottom) and tk_bearish and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below cloud bottom (trend change)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above cloud top (trend change)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0