#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d ADX trend filter and volume confirmation.
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span) to identify trend and momentum.
# ADX > 25 on 1d confirms strong trend, reducing false signals in chop.
# Volume > 1.5x 20-period average adds conviction.
# Trades only when price is above/below cloud with TK cross in direction of trend.
# Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Works in bull markets (trend following) and bear markets (avoids false breaks via ADX filter).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku and ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    chikou = close_1d.copy()  # Will be aligned properly later
    
    # Calculate ADX on 1d
    # +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.append(close_1d[0], close_1d[:-1]))
    tr3 = np.abs(low_1d - np.append(close_1d[0], close_1d[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=1)  # Ichimoku A needs 1 extra bar
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=1)  # Ichimoku B needs 1 extra bar
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou, additional_delay_bars=26)  # Chikou is lagging by 26 periods
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(chikou_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        chikou_val = chikou_aligned[i]
        adx_val = adx_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        # ADX filter: trend strength > 25
        strong_trend = adx_val > 25
        
        # Ichimoku signals
        # Bullish: price above cloud, TK cross bullish (Tenkan > Kijun), Chikou above price 26 periods ago
        price_above_cloud = price > cloud_top
        tk_bullish = tenkan_val > kijun_val
        chikou_above = chikou_val > price  # Chikou compares to current price (already lagged)
        
        # Bearish: price below cloud, TK cross bearish (Tenkan < Kijun), Chikou below price 26 periods ago
        price_below_cloud = price < cloud_bottom
        tk_bearish = tenkan_val < kijun_val
        chikou_below = chikou_val < price
        
        if position == 0:
            # Long conditions: bullish Ichimoku + strong trend + volume confirmation
            if price_above_cloud and tk_bullish and chikou_above and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Ichimoku + strong trend + volume confirmation
            elif price_below_cloud and tk_bearish and chikou_below and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price closes below cloud or TK cross turns bearish
                if price < cloud_bottom or tenkan_val < kijun_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price closes above cloud or TK cross turns bullish
                if price > cloud_top or tenkan_val > kijun_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_1dADX_Volume"
timeframe = "6h"
leverage = 1.0