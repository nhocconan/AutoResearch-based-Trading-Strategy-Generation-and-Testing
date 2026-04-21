#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h Ichimoku Cloud trend filter + Volume Spike
# Long when price breaks above Donchian upper band, price above 12h Kumo cloud, and 12h volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band, price below 12h Kumo cloud, and 12h volume > 1.5x 20-period average
# Exit when price crosses Donchian middle band or volume condition fails
# Ichimoku cloud from 12h provides strong trend filter that works in both bull and bear markets
# Volume spike confirms breakout conviction
# Target: 15-25 trades/year by requiring all three conditions to align

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku components
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Kumo cloud boundaries (use Senkou Span A and B)
    # For trend filter, we use the current cloud (not shifted)
    # The cloud is between Senkou Span A and Senkou Span B
    # We'll use the values without the forward shift for current cloud assessment
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 12h indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Donchian channels on 6h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) - 20 period high/low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        vol_ma = vol_ma_12h_aligned[i]
        
        # Kumo cloud: price above/both spans = bullish, price below/both spans = bearish
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Price above cloud (bullish) or below cloud (bearish)
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        # Get current 12h volume (every 2nd 6h bar since 12h = 2*6h)
        vol_index = i // 2
        if vol_index < len(df_12h):
            volume = df_12h['volume'].iloc[vol_index]
            volume_confirm = volume > 1.5 * vol_ma
        else:
            volume_confirm = False
        
        if position == 0:
            # Long: price breaks above Donchian upper, price above cloud, volume confirmation
            if price > donch_high[i] and price_above_cloud and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, price below cloud, volume confirmation
            elif price < donch_low[i] and price_below_cloud and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian middle or price falls below cloud
                if price < donch_mid[i] or price < cloud_bottom:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian middle or price rises above cloud
                if price > donch_mid[i] or price > cloud_top:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6s_Donchian20_IchimokuCloud_12hVolume"
timeframe = "6h"
leverage = 1.0