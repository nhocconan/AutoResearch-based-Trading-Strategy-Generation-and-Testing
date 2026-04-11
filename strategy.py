#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly Ichimoku cloud and volume confirmation.
# Uses weekly Tenkan-sen/Kijun-sen cross for trend, price above/below cloud for bias,
# and Senkou Span A/B cross for momentum. Volume filter confirms institutional participation.
# Designed for 7-25 trades/year on daily timeframe to minimize fee drag and maximize edge.

name = "1d_1w_ichimoku_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 52 days for weekly calculations
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:  # Need at least 26 weeks for calculations
        return np.zeros(n)
    
    # Weekly Ichimoku calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.array([np.max(high_1w[i-8:i+1]) if i >= 8 else np.nan for i in range(len(high_1w))])
    period9_low = np.array([np.min(low_1w[i-8:i+1]) if i >= 8 else np.nan for i in range(len(low_1w))])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.array([np.max(high_1w[i-25:i+1]) if i >= 25 else np.nan for i in range(len(high_1w))])
    period26_low = np.array([np.min(low_1w[i-25:i+1]) if i >= 25 else np.nan for i in range(len(low_1w))])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.array([np.max(high_1w[i-51:i+1]) if i >= 51 else np.nan for i in range(len(high_1w))])
    period52_low = np.array([np.min(low_1w[i-51:i+1]) if i >= 51 else np.nan for i in range(len(low_1w))])
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.roll(close_1w, 26)
    chikou_span[:26] = np.nan
    
    # Weekly average volume (20-period)
    volume_1w = df_1w['volume'].values
    vol_avg_20 = np.full_like(volume_1w, np.nan, dtype=float)
    for i in range(19, len(volume_1w)):
        vol_avg_20[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align weekly indicators to daily
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1w, chikou_span)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.8 * weekly average volume
        vol_filter = volume[i] > 1.8 * vol_avg_aligned[i]
        
        # Cloud color: green if Senkou Span A > Senkou Span B (bullish), red otherwise
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        cloud_red = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross: Tenkan-sen crosses Kijun-sen
        tk_cross_up = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                       tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_cross_down = (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                         tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        # Kumo twist: Senkou Span A crosses Senkou Span B (momentum shift)
        kumo_twist_up = (senkou_span_a_aligned[i] > senkou_span_b_aligned[i] and 
                         senkou_span_a_aligned[i-1] <= senkou_span_b_aligned[i-1])
        kumo_twist_down = (senkou_span_a_aligned[i] < senkou_span_b_aligned[i] and 
                           senkou_span_a_aligned[i-1] >= senkou_span_b_aligned[i-1])
        
        # Chikou confirmation: Chikou Span above/below price 26 periods ago
        chikou_above_price = chikou_span_aligned[i] > close[i-26] if i >= 26 else False
        chikou_below_price = chikou_span_aligned[i] < close[i-26] if i >= 26 else False
        
        # Long conditions: bullish cloud + TK cross up + price above cloud + Chikou confirmation
        long_condition = (cloud_green and tk_cross_up and price_above_cloud and 
                         chikou_above_price and vol_filter)
        
        # Short conditions: bearish cloud + TK cross down + price below cloud + Chikou confirmation
        short_condition = (cloud_red and tk_cross_down and price_below_cloud and 
                          chikou_below_price and vol_filter)
        
        # Exit conditions: TK cross in opposite direction or price enters opposite cloud
        exit_long = (tk_cross_down or 
                    (position == 1 and close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])))
        exit_short = (tk_cross_up or 
                     (position == -1 and close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])))
        
        # Entry logic
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals