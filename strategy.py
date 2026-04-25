#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrend
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross signals aligned with weekly trend (price above/below weekly Kumo cloud) capture medium-term momentum while avoiding counter-trend whipsaws. Uses volume confirmation (>1.5x 20-bar average) to ensure institutional participation. Discrete sizing (0.25) limits drawdown. Designed for 6h charts to achieve ~12-30 trades/year by requiring confluence of TK cross, weekly trend filter, and volume spike. Works in both bull and bear markets via weekly trend filter that adapts to higher timeframe structure.
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
    
    # Get 1w data for HTF trend filter (weekly Kumo cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1w = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1w = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1w = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (completed weekly bars only)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 9, 20)  # Ichimoku periods and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or 
            np.isnan(senkou_b_1w_aligned[i]) or 
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
        tenkan_val = tenkan_1w_aligned[i]
        kijun_val = kijun_1w_aligned[i]
        senkou_a_val = senkou_a_1w_aligned[i]
        senkou_b_val = senkou_b_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Determine weekly trend: price relative to Kumo cloud
        # Kumo cloud top/bottom (Senkou Span A/B)
        kumo_top = max(senkou_a_val, senkou_b_val)
        kumo_bottom = min(senkou_a_val, senkou_b_val)
        
        # Weekly trend filter: price above cloud = uptrend, below cloud = downtrend
        price_above_kumo = close_val > kumo_top
        price_below_kumo = close_val < kumo_bottom
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = vol_val > 1.5 * vol_ma_val
        
        # TK Cross signals
        # Bullish TK cross: Tenkan crosses above Kijun
        bullish_tk_cross = (tenkan_val > kijun_val) and (tenkan_1w_aligned[i-1] <= kijun_1w_aligned[i-1])
        # Bearish TK cross: Tenkan crosses below Kijun
        bearish_tk_cross = (tenkan_val < kijun_val) and (tenkan_1w_aligned[i-1] >= kijun_1w_aligned[i-1])
        
        if position == 0:
            # Look for entry signals: TK cross aligned with weekly trend and volume confirmation
            # Long: bullish TK cross + price above weekly Kumo (uptrend) + volume spike
            long_signal = bullish_tk_cross and price_above_kumo and volume_spike
            # Short: bearish TK cross + price below weekly Kumo (downtrend) + volume spike
            short_signal = bearish_tk_cross and price_below_kumo and volume_spike
            
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
            # 1. Bearish TK cross (exit long on counter-trend signal)
            if bearish_tk_cross:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Price breaks below weekly Kumo cloud (trend change)
            elif close_val < kumo_bottom:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Bullish TK cross (exit short on counter-trend signal)
            if bullish_tk_cross:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Price breaks above weekly Kumo cloud (trend change)
            elif close_val > kumo_top:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend"
timeframe = "6h"
leverage = 1.0