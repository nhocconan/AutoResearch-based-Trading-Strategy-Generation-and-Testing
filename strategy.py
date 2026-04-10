#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud strategy with 1d/1w trend filter
# - Uses Ichimoku components (Tenkan, Kijun, Senkou Span A/B, Chikou) on 6h
# - Long when price > Cloud AND Tenkan > Kijun AND Chikou > price 26 periods ago
# - Short when price < Cloud AND Tenkan < Kijun AND Chikou < price 26 periods ago
# - 1d/1w ADX > 25 ensures we only trade in strong trends (avoids whipsaws in ranging markets)
# - Ichimoku provides dynamic support/resistance and trend direction
# - Higher timeframe ADX filter prevents counter-trend trades during weak momentum
# - Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# - Discrete position sizing 0.25 to minimize fee churn

name = "6h_1d_1w_ichimoku_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    chikou_shift = 26
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou = np.roll(close, -chikou_shift)  # Negative shift for future values
    
    # Determine Cloud (Kumo) boundaries
    # Senkou Span A and B shifted forward by kijun_period (26)
    senkou_span_a_shifted = np.roll(senkou_span_a, kijun_period)
    senkou_span_b_shifted = np.roll(senkou_span_b, kijun_period)
    # First kijun_period values will be NaN due to roll
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    cloud_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # Pre-compute 1d and 1w ADX for trend filter
    def calculate_dmi(high_arr, low_arr, close_arr, period=14):
        # +DM and -DM
        up_move = np.diff(high_arr, prepend=high_arr[0])
        down_move = np.diff(low_arr, prepend=low_arr[0]) * -1  # Invert to get positive values
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(np.roll(high_arr, 1) - close_arr)
        tr3 = np.abs(np.roll(low_arr, 1) - close_arr)
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smoothing(arr, period):
            result = np.zeros_like(arr)
            result[period-1] = np.mean(arr[1:period+1])  # First value is simple average
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
        minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 
                      0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    # Calculate ADX for 1d and 1w
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_dmi(high_1d, low_1d, close_1d, 14)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_dmi(high_1w, low_1w, close_1w, 14)
    
    # Trend filter: ADX > 25 on both 1d and 1w indicates strong trend
    strong_trend_1d = adx_1d > 25
    strong_trend_1w = adx_1w > 25
    strong_trend = strong_trend_1d & strong_trend_1w
    
    # Align HTF indicators to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup for Ichimoku
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(chikou[i + chikou_shift] if i + chikou_shift < n else np.nan) or
            np.isnan(strong_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get Chikou Span value (current close plotted 26 periods ago)
        chikou_val = chikou[i] if i < len(chikou) else np.nan
        if np.isnan(chikou_val):
            chikou_val = close[i - chikou_shift] if i >= chikou_shift else np.nan
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price above cloud, Tenkan > Kijun, Chikou > price (26 periods ago), strong trend
            if (close[i] > cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                not np.isnan(chikou_val) and 
                chikou_val > close[i - chikou_shift] if i >= chikou_shift else False and
                strong_trend_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price below cloud, Tenkan < Kijun, Chikou < price (26 periods ago), strong trend
            elif (close[i] < cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  not np.isnan(chikou_val) and 
                  chikou_val < close[i - chikou_shift] if i >= chikou_shift else False and
                  strong_trend_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: 
            # For long: price closes below cloud OR Tenkan < Kijun
            # For short: price closes above cloud OR Tenkan > Kijun
            exit_long = (position == 1 and 
                        (close[i] < cloud_bottom[i] or tenkan[i] < kijun[i]))
            exit_short = (position == -1 and 
                         (close[i] > cloud_top[i] or tenkan[i] > kijun[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals