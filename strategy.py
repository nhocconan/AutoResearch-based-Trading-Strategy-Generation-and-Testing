#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Range_MeanReversion
Hypothesis: Weekly pivot points (PP, R1, S1) act as strong support/resistance. Price tends to revert to the weekly PP after touching R1 or S1, especially in ranging markets. 
Uses 1d timeframe with 1h for weekly pivot calculation (via mtf_data). 
Entry: Mean reversion when price touches R1/S1 and closes back inside the weekly range (PP ± 0.5*(R1-S1)) with volume confirmation. 
Exit: When price reaches the opposite boundary or weekly PP. 
Position size 0.25 targets ~15-25 trades/year. Works in ranging markets and avoids strong trends via ADX filter (<25).
"""

name = "1d_Weekly_Pivot_Range_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = PP + (H-L), S1 = PP - (H-L)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_range = h_1w - l_1w
    weekly_r1 = weekly_pp + weekly_range
    weekly_s1 = weekly_pp - weekly_range
    
    # Define entry/exit zones: PP ± 0.5*range for mean reversion
    weekly_range_half = weekly_range * 0.5
    weekly_upper = weekly_pp + weekly_range_half  # PP + 0.5*range
    weekly_lower = weekly_pp - weekly_range_half  # PP - 0.5*range
    
    # Align weekly levels to daily chart (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    
    # ADX filter for ranging markets (ADX < 25)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        atr[period-1] = np.mean(tr[1:period]) if period <= len(tr) else 0
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period]) if period <= len(plus_dm) else 0
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period]) if period <= len(minus_dm) else 0
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if (2*period-1) <= len(dx) else 0
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    range_filter = adx < 25  # Ranging market
    
    # Volume confirmation: current volume > 1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        if position == 0:
            # LONG: Price touches or goes below weekly lower bound and reverts back inside
            if (low[i] <= weekly_lower_aligned[i] and 
                close[i] > weekly_lower_aligned[i] and 
                close[i] < weekly_pp_aligned[i] and  # Still below PP for mean reversion to PP
                range_filter[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above weekly upper bound and reverts back inside
            elif (high[i] >= weekly_upper_aligned[i] and 
                  close[i] < weekly_upper_aligned[i] and 
                  close[i] > weekly_pp_aligned[i] and  # Still above PP for mean reversion to PP
                  range_filter[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly PP or upper bound
            if (close[i] >= weekly_pp_aligned[i]) or \
               (close[i] >= weekly_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly PP or lower bound
            if (close[i] <= weekly_pp_aligned[i]) or \
               (close[i] <= weekly_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals