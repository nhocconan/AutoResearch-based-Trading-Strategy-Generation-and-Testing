#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d/1w trend alignment and volume confirmation
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h for structure
# Only takes breakouts when 1d price is above/below weekly pivot (trend filter)
# Volume confirmation ensures breakout validity
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Works in both bull/bear: 1d/1w trend filter ensures we trade with higher timeframe momentum
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag

name = "6h_1d_1w_ichimoku_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for weekly pivot trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full(len(df_6h), np.nan)
    kijun_sen = np.full(len(df_6h), np.nan)
    senkou_span_a = np.full(len(df_6h), np.nan)
    senkou_span_b = np.full(len(df_6h), np.nan)
    chikou_span = np.full(len(df_6h), np.nan)
    
    for i in range(len(df_6h)):
        if i < 9:
            tenkan_sen[i] = np.nan
            kijun_sen[i] = np.nan
        else:
            tenkan_sen[i] = (df_6h['high'].iloc[i-8:i+1].max() + df_6h['low'].iloc[i-8:i+1].min()) / 2
            
        if i < 26:
            kijun_sen[i] = np.nan
        else:
            kijun_sen[i] = (df_6h['high'].iloc[i-25:i+1].max() + df_6h['low'].iloc[i-25:i+1].min()) / 2
            
        if i >= 26:
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
        else:
            senkou_span_a[i] = np.nan
            
        if i < 52:
            senkou_span_b[i] = np.nan
        else:
            senkou_span_b[i] = (df_6h['high'].iloc[i-51:i+1].max() + df_6h['low'].iloc[i-51:i+1].min()) / 2
            
        # Chikou span (Lagging Span): current close plotted 26 periods back
        if i >= 26:
            chikou_span[i] = df_6h['close'].iloc[i-26]
        else:
            chikou_span[i] = np.nan
    
    # Calculate weekly pivot points from 1w data (using previous week's OHLC)
    weekly_pivot = np.full(len(df_1w), np.nan)
    weekly_r1 = np.full(len(df_1w), np.nan)
    weekly_s1 = np.full(len(df_1w), np.nan)
    weekly_r2 = np.full(len(df_1w), np.nan)
    weekly_s2 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i < 1:
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
        else:
            # Use previous week's OHLC to calculate current week's pivot levels
            prev_high = df_1w['high'].iloc[i-1]
            prev_low = df_1w['low'].iloc[i-1]
            prev_close = df_1w['close'].iloc[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - prev_low
            weekly_s1[i] = 2 * pivot - prev_high
            weekly_r2[i] = pivot + (prev_high - prev_low)
            weekly_s2[i] = pivot - (prev_high - prev_low)
    
    # Align Ichimoku components to 6h timeframe (already in 6h, but need to shift for look-ahead safety)
    # For Ichimoku, we need to ensure we don't use future data - the calculation already uses only historical data
    # However, we need to shift Senkou Span A/B forward by 26 periods (they are plotted ahead)
    # But for our logic, we'll use the current cloud (Senkou Span A/B from 26 periods ago)
    # So we need to access Senkou Span A/B values from 26 periods ago
    
    # Create shifted versions for cloud (Senkou Span A/B plotted 26 periods ahead)
    senkou_span_a_shifted = np.full(len(df_6h), np.nan)
    senkou_span_b_shifted = np.full(len(df_6h), np.nan)
    
    for i in range(len(df_6h)):
        if i >= 26:
            senkou_span_a_shifted[i] = senkou_span_a[i-26]
            senkou_span_b_shifted[i] = senkou_span_b[i-26]
        else:
            senkou_span_a_shifted[i] = np.nan
            senkou_span_b_shifted[i] = np.nan
    
    # Align weekly pivot to 1d timeframe, then to 6h
    weekly_pivot_1d = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup (52 for Senkou Span B)
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a_shifted[i]) or 
            np.isnan(senkou_span_b_shifted[i]) or 
            np.isnan(chikou_span[i]) or 
            np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Determine Ichimoku signal
        # Bullish: price above cloud, Tenkan-sen > Kijun-sen, Chikou span above price 26 periods ago
        # Bearish: price below cloud, Tenkan-sen < Kijun-sen, Chikou span below price 26 periods ago
        price = close[i]
        upper_cloud = max(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        lower_cloud = min(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        
        # Cloud twist: Senkou Span A > Senkou Span B = bullish cloud, A < B = bearish cloud
        cloud_bullish = senkou_span_a_shifted[i] > senkou_span_b_shifted[i]
        
        tenkan_above_kijun = tenkan_sen[i] > kijun_sen[i]
        tenkan_below_kijun = tenkan_sen[i] < kijun_sen[i]
        
        # Chikou span confirmation: compare current Chikou span to price 26 periods ago
        chikou_confirm_bullish = chikou_span[i] > close[i-26] if i >= 26 else False
        chikou_confirm_bearish = chikou_span[i] < close[i-26] if i >= 26 else False
        
        # Ichimoku bullish signal: price above cloud + Tenkan > Kijun + Chikou bullish
        ichimoku_bullish = (price > upper_cloud and 
                           tenkan_above_kijun and 
                           chikou_confirm_bullish)
        
        # Ichimoku bearish signal: price below cloud + Tenkan < Kijun + Chikou bearish
        ichimoku_bearish = (price < lower_cloud and 
                           tenkan_below_kijun and 
                           chikou_confirm_bearish)
        
        # Weekly pivot trend filter: only trade long when price above weekly pivot, short when below
        price_above_weekly_pivot = price > weekly_pivot_6h[i]
        price_below_weekly_pivot = price < weekly_pivot_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions: Ichimoku turn bearish OR price breaks below weekly pivot
            if ichimoku_bearish or not price_above_weekly_pivot:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Ichimoku turn bullish OR price breaks above weekly pivot
            if ichimoku_bullish or not price_below_weekly_pivot:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Ichimoku signal with volume confirmation and weekly pivot trend filter
            if volume_confirm:
                # Long entry: Ichimoku bullish + price above weekly pivot
                if ichimoku_bullish and price_above_weekly_pivot:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Ichimoku bearish + price below weekly pivot
                elif ichimoku_bearish and price_below_weekly_pivot:
                    position = -1
                    signals[i] = -0.25
    
    return signals