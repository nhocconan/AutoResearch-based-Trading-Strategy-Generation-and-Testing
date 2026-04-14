#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation
# Long when Tenkan-sen crosses above Kijun-sen, price is above Kumo (cloud), and Senkou Span A > Senkou Span B (bullish cloud)
# Short when Tenkan-sen crosses below Kijun-sen, price is below Kumo, and Senkou Span A < Senkou Span B (bearish cloud)
# Volume must be > 1.5x 20-period average for confirmation
# Uses 1-day Ichimoku for trend filter: only take long if price > 1-day Senkou Span A, short if price < 1-day Senkou Span B
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h Ichimoku components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Calculate 6h volume average (20-period)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1-day Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-day Senkou Span B (for trend filter)
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 52-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_6h_current = volume[i]
        
        # Bullish conditions: Tenkan > Kijun, price above cloud, bullish cloud
        bullish_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        price_above_cloud = price > senkou_span_a_aligned[i] and price > senkou_span_b_aligned[i]
        bullish_cloud = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        
        # Bearish conditions: Tenkan < Kijun, price below cloud, bearish cloud
        bearish_cross = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        price_below_cloud = price < senkou_span_a_aligned[i] and price < senkou_span_b_aligned[i]
        bearish_cloud = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Volume confirmation
        volume_confirmed = vol_6h_current > 1.5 * vol_ma_6h_aligned[i]
        
        # 1-day trend filter: only long if price > 1-day Senkou Span B, short if price < 1-day Senkou Span B
        trend_filter_long = price > senkou_span_b_1d_aligned[i]
        trend_filter_short = price < senkou_span_b_1d_aligned[i]
        
        if position == 0:
            # Long setup
            if (bullish_cross and price_above_cloud and bullish_cloud and 
                volume_confirmed and trend_filter_long):
                position = 1
                signals[i] = position_size
            # Short setup
            elif (bearish_cross and price_below_cloud and bearish_cloud and 
                  volume_confirmed and trend_filter_short):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price drops below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                price < senkou_span_a_aligned[i] or price < senkou_span_b_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                price > senkou_span_a_aligned[i] or price > senkou_span_b_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_1dTrendFilter_Volume"
timeframe = "6h"
leverage = 1.0