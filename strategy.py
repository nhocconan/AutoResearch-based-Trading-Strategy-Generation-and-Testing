#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d Weekly Pivot + Volume Confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Ichimoku provides trend, momentum, and support/resistance in one system
# Weekly pivot from 1d HTF determines institutional bias (long above weekly pivot, short below)
# Volume spike (1.8x 20-period average) confirms breakout/continuation validity
# Works in bull markets via cloud breakouts with trend alignment and bear markets via cloud rejection
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_IchimokuCloud_1dWeeklyPivot_VolumeSpike"
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
    
    # Calculate 1d Ichimoku components (prior completed 1d bar's values)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().shift(1).values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().shift(1).values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().shift(1).values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().shift(1).values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().shift(1).values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().shift(1).values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 1d Weekly Pivot (from prior week's daily data)
    # We need to aggregate daily data to weekly - using the same 1d data but resampling conceptually
    # Since we can't resample, we'll use a 5-day approximation for weekly pivot
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly high, low, close from last 5 daily bars (approximation)
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly Pivot Point: (High + Low + Close)/3
    weekly_pivot = (week_high + week_low + week_close) / 3
    # Weekly R1: (2*Pivot) - Low
    weekly_r1 = 2 * weekly_pivot - week_low
    # Weekly S1: (2*Pivot) - High
    weekly_s1 = 2 * weekly_pivot - week_high
    # Weekly R2: Pivot + (High - Low)
    weekly_r2 = weekly_pivot + (week_high - week_low)
    # Weekly S2: Pivot - (High - Low)
    weekly_s2 = weekly_pivot - (week_high - week_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Calculate 6h volume spike (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price above cloud AND tenkan > kijun (bullish momentum) AND price > weekly pivot AND volume spike
            if (close[i] > cloud_top and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud AND tenkan < kijun (bearish momentum) AND price < weekly pivot AND volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below cloud OR tenkan < kijun (momentum change) OR price < weekly pivot
            if (close[i] < cloud_bottom or 
                tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR tenkan > kijun (momentum change) OR price > weekly pivot
            if (close[i] > cloud_top or 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals