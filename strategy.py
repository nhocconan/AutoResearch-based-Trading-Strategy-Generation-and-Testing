#!/usr/bin/env python3
"""
Experiment #7647: 6h Ichimoku Cloud with 1-day trend filter and volume confirmation.
Hypothesis: Use daily Ichimoku cloud as trend filter (price above/below cloud) and
Tenkan/Kijun cross on 6h for entry timing. Volume must exceed 1.5x average to confirm.
In bull trend (price > daily cloud), go long on TK cross up. In bear trend (price < daily cloud),
go short on TK cross down. Targets 50-150 trades over 4 years (12-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7647_6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Tenkan-sen (Conversion Line)
KIJUN_PERIOD = 26    # Kijun-sen (Base Line)
SENKOU_SPAN_B_PERIOD = 52  # Senkou Span B
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A: (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B: (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                     pd.Series(low_1d).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2
    
    # Future Kumo (cloud) data: shift Senkou spans forward by 26 periods
    senkou_span_a_shifted = senkou_span_a.shift(26)
    senkou_span_b_shifted = senkou_span_b.shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted.values)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted.values)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen and Kijun-sen
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, VOLUME_MA_PERIOD) + 30  # +30 for Ichimoku shift
    
    for i in range(start, n):
        # Skip if Ichimoku data not available
        if (np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        lower_cloud = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Price above/below cloud determines trend
        bull_trend = close[i] > upper_cloud
        bear_trend = close[i] < lower_cloud
        
        # TK cross conditions (need previous values)
        if i < 1:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        tk_cross_up = (tenkan_6h[i-1] <= kijun_6h[i-1]) and (tenkan_6h[i] > kijun_6h[i])
        tk_cross_down = (tenkan_6h[i-1] >= kijun_6h[i-1]) and (tenkan_6h[i] < kijun_6h[i])
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_trend and tk_cross_up and volume_confirmed
        short_entry = bear_trend and tk_cross_down and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit on TK cross down or price enters cloud
            if tk_cross_down or (close[i] >= lower_cloud and close[i] <= upper_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit on TK cross up or price enters cloud
            if tk_cross_up or (close[i] >= lower_cloud and close[i] <= upper_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals