#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Ichimoku cloud direction with 1d Tenkan/Kijun cross for entry and volume confirmation
# Long when price is above weekly Ichimoku cloud, 1d Tenkan crosses above Kijun, and volume > 1.5x average
# Short when price is below weekly Ichimoku cloud, 1d Tenkan crosses below Kijun, and volume > 1.5x average
# Weekly Ichimoku provides strong trend filter, daily TK cross provides timely entry, volume confirms strength.
# Works in bull/bear markets by only taking trades in direction of higher timeframe trend.
# Target: 12-37 trades per year (50-150 over 4 years) with 0.25 position sizing.

name = "6h_1wIchimoku_1dTKCross_Volume_v1"
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
    
    # Calculate weekly Ichimoku components ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min()
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min()
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Calculate 1-day Tenkan/Kijun for entry signal
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # 1-day Tenkan-sen (9-period)
    td_period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    td_period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    td_tenkan = ((td_period9_high + td_period9_low) / 2).values
    
    # 1-day Kijun-sen (26-period)
    td_period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    td_period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    td_kijun = ((td_period26_high + td_period26_low) / 2).values
    
    # Align 1-day TK to 6h timeframe
    td_tenkan_aligned = align_htf_to_ltf(prices, df_1d, td_tenkan)
    td_kijun_aligned = align_htf_to_ltf(prices, df_1d, td_kijun)
    
    # TK cross signals: 1 for bullish cross, -1 for bearish cross, 0 otherwise
    tk_cross = np.zeros(n)
    tk_cross[1:] = np.where(
        (td_tenkan_aligned[1:] > td_kijun_aligned[1:]) & 
        (td_tenkan_aligned[:-1] <= td_kijun_aligned[:-1]), 1,
        np.where(
            (td_tenkan_aligned[1:] < td_kijun_aligned[1:]) & 
            (td_tenkan_aligned[:-1] >= td_kijun_aligned[:-1]), -1, 0
        )
    )
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(td_tenkan_aligned[i]) or np.isnan(td_kijun_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below weekly cloud
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: price above cloud, bullish TK cross, volume confirmation
            if price_above_cloud and tk_cross[i] == 1 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, bearish TK cross, volume confirmation
            elif price_below_cloud and tk_cross[i] == -1 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below cloud or bearish TK cross
            if close[i] < cloud_top or tk_cross[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above cloud or bullish TK cross
            if close[i] > cloud_bottom or tk_cross[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals