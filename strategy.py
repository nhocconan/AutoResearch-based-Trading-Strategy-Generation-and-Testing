#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Ichimoku cloud filter with 1d TK cross for entry timing and volume spike confirmation
# Long when price > 12h Ichimoku cloud (bullish) AND 1d Tenkan-sen crosses above Kijun-sen (bullish TK cross) AND volume > 1.8 * avg_volume(20) on 6h
# Short when price < 12h Ichimoku cloud (bearish) AND 1d Tenkan-sen crosses below Kijun-sen (bearish TK cross) AND volume > 1.8 * avg_volume(20) on 6h
# Exit when price crosses back into the 12h Ichimoku cloud (mean reversion to cloud boundary)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Ichimoku cloud provides dynamic support/resistance and trend filter
# 1d TK cross provides precise entry timing with lower lag
# Volume spike confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy cloud breakouts) and bear (sell cloud breakdowns) markets

name = "6h_12hIchimoku_CloudFilter_1dTKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Ichimoku cloud calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # Need at least 52 completed 12h bars for Ichimoku (26*2)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For trend identification: price above cloud = bullish, below cloud = bearish
    # We need to align the cloud values to current time (no look-ahead)
    # Since Senkou Span is plotted 26 periods ahead, we use current values for cloud edges
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align 12h Ichimoku cloud to 6h timeframe (wait for completed 12h bar)
    cloud_top_aligned = align_htf_to_ltf(prices, df_12h, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_12h, cloud_bottom)
    
    # Get 1d data ONCE before loop for TK cross calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 completed 1d bars for Kijun-sen
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku Tenkan-sen and Kijun-sen for TK cross
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Calculate TK cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_above = (tenkan_sen_1d > kijun_sen_1d) & (tenkan_sen_1d <= kijun_sen_1d)  # Just crossed above
    tk_cross_below = (tenkan_sen_1d < kijun_sen_1d) & (tenkan_sen_1d >= kijun_sen_1d)  # Just crossed below
    # Fix the logic: need to check previous bar
    tk_cross_above = (tenkan_sen_1d > kijun_sen_1d) & (np.roll(tenkan_sen_1d, 1) <= np.roll(kijun_sen_1d, 1))
    tk_cross_below = (tenkan_sen_1d < kijun_sen_1d) & (np.roll(tenkan_sen_1d, 1) >= np.roll(kijun_sen_1d, 1))
    # Handle first bar
    tk_cross_above[0] = False
    tk_cross_below[0] = False
    
    # Align 1d TK cross to 6h timeframe (wait for completed 1d bar)
    tk_cross_above_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_above.astype(float))
    tk_cross_below_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_below.astype(float))
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(tk_cross_above_aligned[i]) or np.isnan(tk_cross_below_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > 12h cloud (bullish) AND 1d TK cross bullish AND volume spike, in session
            if (close[i] > cloud_top_aligned[i] and 
                tk_cross_above_aligned[i] > 0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < 12h cloud (bearish) AND 1d TK cross bearish AND volume spike, in session
            elif (close[i] < cloud_bottom_aligned[i] and 
                  tk_cross_below_aligned[i] > 0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h cloud top (mean reversion to cloud)
            if close[i] < cloud_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h cloud bottom (mean reversion to cloud)
            if close[i] > cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals