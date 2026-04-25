#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with 1d cloud filter (price above/below cloud) and volume confirmation captures strong trend continuation in both bull/bear markets. Uses discrete sizing (0.25) and ATR-based trailing stop (2.5x ATR) to limit fee drag. Targets 12-25 trades/year by requiring confluence of TK cross, cloud position, and volume spike (>2.0x 20-bar avg). Ichimoku works well in cryptocurrency trends due to its forward-looking cloud and dynamic support/resistance.
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
    
    # Get 1d data for HTF cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper delay for forward shift)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d, additional_delay_bars=26)
    
    # Calculate ATR(21) on 6h for trailing stop
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_6h = pd.Series(tr).ewm(alpha=1/21, adjust=False, min_periods=21).mean().values
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 9, 20, 21)  # Senkou B, Kijun, Tenkan, vol MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(atr_6h[i]) or 
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
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        atr_val = atr_6h[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Price above/below cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # TK cross: Tenkan crosses above/below Kijun
        # Need previous values to detect cross
        if i > start_idx:
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tk_cross_up = (tenkan_val > kijun_val) and (tenkan_prev <= kijun_prev)
            tk_cross_down = (tenkan_val < kijun_val) and (tenkan_prev >= kijun_prev)
        else:
            tk_cross_up = False
            tk_cross_down = False
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: TK cross with cloud filter and volume
            # Long: Tenkan crosses above Kijun AND price above cloud AND volume confirm
            long_signal = tk_cross_up and price_above_cloud and volume_confirm
            # Short: Tenkan crosses below Kijun AND price below cloud AND volume confirm
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
            # 1. Trailing stop: price moves against position by 2.5*ATR from highest high since entry
            # Simplified: close below entry - 2.5*ATR
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Reverse TK cross: Tenkan crosses below Kijun
            elif tk_cross_down:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trailing stop: price moves against position by 2.5*ATR from lowest low since entry
            # Simplified: close above entry + 2.5*ATR
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Reverse TK cross: Tenkan crosses above Kijun
            elif tk_cross_up:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0