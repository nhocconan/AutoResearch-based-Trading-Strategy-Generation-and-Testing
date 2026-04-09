#!/usr/bin/env python3
# 12h_daily_pivot_breakout_volume_v1
# Hypothesis: 12h strategy using daily pivot levels (standard, not Camarilla) with volume confirmation and trend filter.
# Long: Price breaks above R1 with volume > 1.5x 20-period average and 12h close > 12h EMA20.
# Short: Price breaks below S1 with volume > 1.5x 20-period average and 12h close < 12h EMA20.
# Exit: Price returns to pivot point (PP) for both long and short.
# Uses standard pivot calculation: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
# Standard pivots are widely watched and work in ranging markets (common in 2025 bear) while capturing breakouts in trends.
# 12h timeframe reduces noise and overtrading vs lower timeframes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_pivot_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for standard pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate standard pivot points for each 1d bar
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1: R1 = 2*PP - Low
    r1 = 2 * pp - low_1d
    # Support 1: S1 = 2*PP - High
    s1 = 2 * pp - high_1d
    
    # Align pivot levels to 12h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA20 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema_20_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA20 to 12h (no shift needed as both are 12h)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # 12h trend filter: close > EMA20 for uptrend, < EMA20 for downtrend
        trend_12h_up = close_12h_aligned[i] > ema_20_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above R1 with volume and 12h uptrend
            if (close[i] > r1_aligned[i] and    # Break above R1
                volume_confirmed and           # Volume spike
                trend_12h_up):                 # 12h uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S1 with volume and 12h downtrend
            elif (close[i] < s1_aligned[i] and  # Break below S1
                  volume_confirmed and          # Volume spike
                  trend_12h_down):              # 12h downtrend
                position = -1
                signals[i] = -0.25
    
    return signals