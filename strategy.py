#!/usr/bin/env python3
name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close_1d).shift(26)
    
    # Align to 6h timeframe (wait for 26-period shift)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values, additional_delay_bars=26)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values, additional_delay_bars=26)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values, additional_delay_bars=26)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span.values, additional_delay_bars=0)  # already lagged
    
    # Cloud: future Senkou Span A/B
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(chikou_span_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + Chikou above price 26 periods ago + volume
            tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            price_above_cloud = close[i] > upper_cloud[i]
            chikou_confirm = chikou_span_aligned[i] > close[i - 26] if i >= 26 else False
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            
            if tk_bullish and price_above_cloud and chikou_confirm and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + Chikou below price 26 periods ago + volume
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < lower_cloud[i] and 
                  chikou_span_aligned[i] < close[i - 26] if i >= 26 else False and
                  volume[i] > vol_ma_4[i] * 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish or price drops below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                close[i] < lower_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish or price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                close[i] > upper_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku TK cross with cloud filter and Chikou confirmation
# - TK cross (Tenkan-sen/Kijun-sen) signals momentum shift
# - Trade only when price is above/below cloud (trend filter)
# - Chikou Span confirms trend by comparing current price to price 26 periods ago
# - Volume spike (2x average) filters for institutional participation
# - Works in both bull (buy TK cross bullish above cloud) and bear (sell TK cross bearish below cloud)
# - Exit when TK cross reverses or price re-enters cloud
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Uses daily Ichimoku for higher timeframe context with proper alignment
# - Avoids whipsaws by requiring multiple confirmation layers