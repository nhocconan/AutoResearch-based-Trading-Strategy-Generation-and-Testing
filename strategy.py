#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h timeframe with daily cloud filter (price above/below cloud) and volume confirmation (>1.5x average volume). 
In bull markets: price above cloud + TK cross up = long. 
In bear markets: price below cloud + TK cross down = short. 
The cloud acts as dynamic support/resistance and trend filter, reducing whipsaw in sideways markets. 
Volume confirmation ensures breakouts have conviction. 
Designed to work in both bull (trend-following) and bear (mean-reversion at cloud edges) by requiring alignment with daily Ichimoku cloud.
Targets 12-25 trades/year on 6h timeframe to minimize fee drag while capturing strong momentum moves.
Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
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
    
    # Get 6h data for Ichimoku calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Get 1d data for cloud filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku cloud on 1d (same as above but for daily)
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    senkou_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # The cloud is between Senkou Span A and B
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    lower_cloud_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align all 6h Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Align 1d cloud to 6h timeframe
    upper_cloud_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud_1d)
    lower_cloud_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud_1d)
    
    # Calculate ATR(14) for stoploss on 6h
    # True Range
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_6h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 9, 20)  # Ichimoku needs 52, vol needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(upper_cloud_aligned[i]) or 
            np.isnan(lower_cloud_aligned[i]) or 
            np.isnan(atr_6h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        upper_cloud_val = upper_cloud_aligned[i]
        lower_cloud_val = lower_cloud_aligned[i]
        atr_val = atr_6h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        # Ichimoku signals:
        # TK Cross: Tenkan-sen crossing Kijun-sen
        # We need previous values to detect cross
        if i > start_idx:
            prev_tenkan = tenkan_sen_aligned[i-1]
            prev_kijun = kijun_sen_aligned[i-1]
            tk_cross_up = (prev_tenkan <= prev_kijun) and (tenkan_val > kijun_val)
            tk_cross_down = (prev_tenkan >= prev_kijun) and (tenkan_val < kijun_val)
        else:
            tk_cross_up = False
            tk_cross_down = False
        
        # Price relative to cloud
        price_above_cloud = close_val > upper_cloud_val
        price_below_cloud = close_val < lower_cloud_val
        
        if position == 0:
            # Look for entry signals: TK cross with cloud filter and volume confirmation
            # Long: TK cross up + price above cloud + volume confirmation
            long_signal = tk_cross_up and price_above_cloud and volume_confirm
            # Short: TK cross down + price below cloud + volume confirmation
            short_signal = tk_cross_down and price_below_cloud and volume_confirm
            
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
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. TK cross down (exit long on bearish cross)
            elif tk_cross_down:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Price falls below cloud (exit long if cloud turns bearish)
            elif close_val < lower_cloud_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. TK cross up (exit short on bullish cross)
            elif tk_cross_up:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Price rises above cloud (exit short if cloud turns bullish)
            elif close_val > upper_cloud_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0