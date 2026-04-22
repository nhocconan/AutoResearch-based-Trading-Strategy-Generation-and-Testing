# 6h Ichimoku Cloud Strategy with 1d Trend Filter and Volume Confirmation
# Hypothesis: Uses Ichimoku Cloud on 6h timeframe for trend identification and momentum,
# with 1d Ichimoku cloud as higher timeframe trend filter. Volume spikes confirm
# institutional participation. The strategy avoids counter-trend trades and works in
# both bull and bear markets by aligning with the dominant 1d trend while capturing
# momentum from cloud breaks and TK crosses. Target: 15-30 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    period9_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    period26_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    period52_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku for entry signals
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # Calculate 6h volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Wait for Senkou Span B calculation
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend: price above/below cloud
        # Cloud top is max(senkou_a, senkou_b), bottom is min(senkou_a, senkou_b)
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Determine 6h Ichimoku signals
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        senkou_a_6h_val = senkou_a_6h[i]
        senkou_b_6h_val = senkou_b_6h[i]
        
        # Cloud top/bottom for 6h
        cloud_top_6h = np.maximum(senkou_a_6h_val, senkou_b_6h_val)
        cloud_bottom_6h = np.minimum(senkou_a_6h_val, senkou_b_6h_val)
        
        # TK Cross signals
        tk_cross_bull = tenkan_6h_val > kijun_6h_val
        tk_cross_bear = tenkan_6h_val < kijun_6h_val
        
        # Price relative to cloud
        price_above_cloud_6h = close[i] > cloud_top_6h
        price_below_cloud_6h = close[i] < cloud_bottom_6h
        
        if position == 0:
            # Long: bullish TK cross, price above 6h cloud, price above 1d cloud, volume spike
            if (tk_cross_bull and price_above_cloud_6h and 
                close[i] > cloud_top_1d and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross, price below 6h cloud, price below 1d cloud, volume spike
            elif (tk_cross_bear and price_below_cloud_6h and 
                  close[i] < cloud_bottom_1d and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross in opposite direction or price re-enters cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish TK cross or price below 6h cloud
                if tk_cross_bear or price_below_cloud_6h:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish TK cross or price above 6h cloud
                if tk_cross_bull or price_above_cloud_6h:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0