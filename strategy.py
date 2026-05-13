#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(tenkan).max() + pd.Series(low).rolling(tenkan).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(kijun).max() + pd.Series(low).rolling(kijun).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(senkou).max() + pd.Series(low).rolling(senkou).min()) / 2).shift(kijun)
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1D data ONCE for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on 1D
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1D Ichimoku to 6H timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # 6H price for entry timing
    close_s = pd.Series(close)
    ema20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient data
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(chikou_1d_aligned[i]) or np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Cloud calculation
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Ichimoku signals
        tk_cross_bull = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_bear = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        chikou_confirm_bull = chikou_1d_aligned[i] > close[i - 26] if i >= 26 else False
        chikou_confirm_bear = chikou_1d_aligned[i] < close[i - 26] if i >= 26 else False
        
        # Price relative to 6H EMA for entry timing
        price_above_ema = close[i] > ema20_6h[i]
        price_below_ema = close[i] < ema20_6h[i]
        
        if position == 0:
            # LONG: Bullish TK cross + price above cloud + Chikou confirmation + price above EMA
            if tk_cross_bull and price_above_cloud and chikou_confirm_bull and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish TK cross + price below cloud + Chikou confirmation + price below EMA
            elif tk_cross_bear and price_below_cloud and chikou_confirm_bear and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish TK cross OR price drops below cloud
            if tk_cross_bear or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish TK cross OR price rises above cloud
            if tk_cross_bull or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(tenkan).max() + pd.Series(low).rolling(tenkan).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(kijun).max() + pd.Series(low).rolling(kijun).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(senkou).max() + pd.Series(low).rolling(senkou).min()) / 2).shift(kijun)
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1D data ONCE for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on 1D
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1D Ichimoku to 6H timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # 6H price for entry timing
    close_s = pd.Series(close)
    ema20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient data
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(chikou_1d_aligned[i]) or np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Cloud calculation
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Ichimoku signals
        tk_cross_bull = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_bear = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        chikou_confirm_bull = chikou_1d_aligned[i] > close[i - 26] if i >= 26 else False
        chikou_confirm_bear = chikou_1d_aligned[i] < close[i - 26] if i >= 26 else False
        
        # Price relative to 6H EMA for entry timing
        price_above_ema = close[i] > ema20_6h[i]
        price_below_ema = close[i] < ema20_6h[i]
        
        if position == 0:
            # LONG: Bullish TK cross + price above cloud + Chikou confirmation + price above EMA
            if tk_cross_bull and price_above_cloud and chikou_confirm_bull and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish TK cross + price below cloud + Chikou confirmation + price below EMA
            elif tk_cross_bear and price_below_cloud and chikou_confirm_bear and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish TK cross OR price drops below cloud
            if tk_cross_bear or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish TK cross OR price rises above cloud
            if tk_cross_bull or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals