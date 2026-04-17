#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Ichimoku cloud + Daily price action with volume confirmation
# In bull markets: price above cloud = bullish bias, look for long entries on pullbacks to Tenkan/Kijun
# In bear markets: price below cloud = bearish bias, look for short entries on rallies to Tenkan/Kijun
# Weekly timeframe provides stable trend bias, daily timeframe provides entry timing
# Volume confirmation filters out false breakouts. Target: 15-25 trades/year to avoid fee drag.
# Works in both bull and bear because it follows the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku cloud (primary trend filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for entry signals and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For trend bias, we'll use current price vs cloud
    
    # Align Ichimoku components to daily timeframe
    tenkan_sen_1d = align_htf_to_ltf(df_1d, df_1w, tenkan_sen)
    kijun_sen_1d = align_htf_to_ltf(df_1d, df_1w, kijun_sen)
    senkou_span_a_1d = align_htf_to_ltf(df_1d, df_1w, senkou_span_a)
    senkou_span_b_1d = align_htf_to_ltf(df_1d, df_1w, senkou_span_b)
    
    # Calculate cloud boundaries
    senkou_top = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    senkou_bottom = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    # Determine trend bias: price above/below cloud
    price_above_cloud = close_1d > senkou_top
    price_below_cloud = close_1d < senkou_bottom
    
    # Daily Tenkan/Kijun for entry signals
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_daily = (period9_high_1d + period9_low_1d) / 2.0
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_daily = (period26_high_1d + period26_low_1d) / 2.0
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(len(df_1d))
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(26, 20)  # Need Ichimoku and volume data
    
    for i in range(start_idx, len(df_1d)):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_1d[i]) or np.isnan(kijun_sen_1d[i]) or 
            np.isnan(senkou_top[i]) or np.isnan(senkou_bottom[i]) or
            np.isnan(tenkan_sen_daily[i]) or np.isnan(kijun_sen_daily[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume_1d[i] > (1.3 * volume_ma20[i])
        
        # Price relative to daily Tenkan/Kijun
        price_above_tenkan = close_1d[i] > tenkan_sen_daily[i]
        price_below_tenkan = close_1d[i] < tenkan_sen_daily[i]
        price_above_kijun = close_1d[i] > kijun_sen_daily[i]
        price_below_kijun = close_1d[i] < kijun_sen_daily[i]
        
        if position == 0:
            # Long: price above weekly cloud AND price crosses above daily Tenkan/Kijun with volume
            if (price_above_cloud[i] and price_above_tenkan[i] and price_above_kijun[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly cloud AND price crosses below daily Tenkan/Kijun with volume
            elif (price_below_cloud[i] and price_below_tenkan[i] and price_below_kijun[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below daily Kijun OR price goes below weekly cloud
            if (price_below_kijun[i] or price_below_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above daily Kijun OR price goes above weekly cloud
            if (price_above_kijun[i] or price_above_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    # Align signals to original price index
    # Since we used daily data for signals, we need to map back to original timeframe
    # For daily timeframe strategy, signals are already aligned
    if len(signals) == n:
        return signals
    else:
        # If different length, create zero array and fill where possible
        result = np.zeros(n)
        min_len = min(len(signals), n)
        result[:min_len] = signals[:min_len]
        return result

name = "1d_Ichimoku_Cloud_Trend_Follow"
timeframe = "1d"
leverage = 1.0