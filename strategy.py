#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Ichimoku Cloud breakout with volume confirmation.
# Long when price breaks above Kumo (cloud) top AND weekly Kijun-sen > weekly Tenkan-sen with volume spike (>2x average).
# Short when price breaks below Kumo (cloud) bottom AND weekly Kijun-sen < weekly Tenkan-sen with volume spike.
# Uses weekly Ichimoku as trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 10-25 trades/year per symbol (~40-100 total over 4 years).
name = "1d_IchimokuCloud_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_weekly).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_weekly).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_weekly).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_weekly).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_weekly).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_weekly).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to daily timeframe (wait for weekly close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_weekly, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_weekly, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_b)
    
    # Calculate daily Kumo (cloud) top and bottom
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Need both Ichimoku and volume data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above Kumo top AND weekly Kijun > Tenkan
            if price > kumo_top_val and kijun_val > tenkan_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Kumo bottom AND weekly Kijun < Tenkan
            elif price < kumo_bottom_val and kijun_val < tenkan_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below Kumo bottom or weekly Kijun < Tenkan
            if price < kumo_bottom_val or kijun_val < tenkan_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above Kumo top or weekly Kijun > Tenkan
            if price > kumo_top_val or kijun_val > tenkan_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals