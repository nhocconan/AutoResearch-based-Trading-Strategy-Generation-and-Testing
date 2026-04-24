#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with Weekly Trend Filter and Volume Confirmation
- Primary timeframe: 6h for Ichimoku calculations and entries/exits.
- HTF: 1w Ichimoku cloud color (bullish if Senkou Span A > Senkou Span B) for trend direction.
- Volume: Current 6h volume > 1.8 * 30-period volume MA to avoid low-volume false signals.
- Entry: Long when Tenkan-sen crosses above Kijun-sen AND price is above cloud AND weekly trend bullish AND volume spike.
         Short when Tenkan-sen crosses below Kijun-sen AND price is below cloud AND weekly trend bearish AND volume spike.
- Exit: Opposite Tenkan/Kijun cross OR loss of weekly trend alignment OR loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Ichimoku provides comprehensive trend, momentum, and support/resistance in one system.
The weekly trend filter ensures we only trade in the direction of the higher timeframe trend,
reducing whipsaws during range-bound periods. Volume confirmation avoids low-conviction breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components for 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Ichimoku cloud (already shifted in calculation above)
    # For price vs cloud comparison, we use current Senkou spans
    senkou_a_current = senkou_a  # These are already calculated for current time
    senkou_b_current = senkou_b
    
    # Get 1w data for Ichimoku cloud color (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate 1w Ichimoku components for trend filter
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # 1w Tenkan-sen
    period9_high_1w = pd.Series(df_1w_high).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(df_1w_low).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2
    
    # 1w Kijun-sen
    period26_high_1w = pd.Series(df_1w_high).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(df_1w_low).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2
    
    # 1w Senkou Span A
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # 1w Senkou Span B
    period52_high_1w = pd.Series(df_1w_high).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(df_1w_low).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((period52_high_1w + period52_low_1w) / 2)
    
    # Weekly trend: bullish if Senkou Span A > Senkou Span B
    weekly_trend_bullish = senkou_a_1w > senkou_b_1w
    weekly_trend_bearish = senkou_a_1w < senkou_b_1w
    
    # Align HTF indicators to 6h
    weekly_trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bullish.astype(float))
    weekly_trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bearish.astype(float))
    
    # Volume confirmation: current 6h volume > 1.8 * 30-period volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 30)  # Need enough bars for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_current[i]) or np.isnan(senkou_b_current[i]) or
            np.isnan(weekly_trend_bullish_aligned[i]) or np.isnan(weekly_trend_bearish_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        prev_tenkan = tenkan[i-1] if i > 0 else curr_tenkan
        prev_kijun = kijun[i-1] if i > 0 else curr_kijun
        weekly_bull = weekly_trend_bullish_aligned[i] > 0.5
        weekly_bear = weekly_trend_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Tenkan crosses above Kijun AND price above cloud AND weekly bullish
                if (prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun and 
                    curr_close > max(senkou_a_current[i], senkou_b_current[i]) and weekly_bull):
                    signals[i] = 0.25
                    position = 1
                # Bearish: Tenkan crosses below Kijun AND price below cloud AND weekly bearish
                elif (prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun and 
                      curr_close < min(senkou_a_current[i], senkou_b_current[i]) and weekly_bear):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price falls below cloud OR weekly turns bearish OR volume drops
            tenkan_cross_down = (prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun)
            price_below_cloud = curr_close < min(senkou_a_current[i], senkou_b_current[i])
            weekly_turn_bear = weekly_trend_bearish_aligned[i] > 0.5
            
            if tenkan_cross_down or price_below_cloud or weekly_turn_bear or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud OR weekly turns bullish OR volume drops
            tenkan_cross_up = (prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun)
            price_above_cloud = curr_close > max(senkou_a_current[i], senkou_b_current[i])
            weekly_turn_bull = weekly_trend_bullish_aligned[i] > 0.5
            
            if tenkan_cross_up or price_above_cloud or weekly_turn_bull or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0