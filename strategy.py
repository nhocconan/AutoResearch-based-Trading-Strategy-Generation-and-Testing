#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: 4h Camarilla R1/S1 breakout filtered by 12h EMA50 trend and volume surge.
# Camarilla levels provide high-probability reversal/breakout points.
# Breakout above R1 or below S1 with volume and trend continuation captures strong moves.
# Works in bull/bear markets by using 12h trend filter and requiring volume confirmation.
# Targets 20-50 trades/year to minimize fee drag on 4h timeframe.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: Close +- (High - Low) * multipliers
    # R1 = Close + (High - Low) * 1.0833
    # S1 = Close - (High - Low) * 1.0833
    # R2 = Close + (High - Low) * 1.1666
    # S2 = Close - (High - Low) * 1.1666
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    camarilla_multiplier = 1.0833  # for R1/S1
    high_low_range = high_1d - low_1d
    
    r1 = close_1d_vals + (high_low_range * camarilla_multiplier)
    s1 = close_1d_vals - (high_low_range * camarilla_multiplier)
    r2 = close_1d_vals + (high_low_range * 1.1666)
    s2 = close_1d_vals - (high_low_range * 1.1666)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) + volume MA (30)
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA50
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_50_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_r1 = close[i] > r1_aligned[i-1]
        breakdown_s1 = close[i] < s1_aligned[i-1]
        breakout_r2 = close[i] > r2_aligned[i-1]
        breakdown_s2 = close[i] < s2_aligned[i-1]
        
        if position == 0:
            # Long: Camarilla R1 breakout with volume surge and 12h uptrend
            if breakout_r1 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S1 breakdown with volume surge and 12h downtrend
            elif breakdown_s1 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if close[i] < s1_aligned[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if close[i] > r1_aligned[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals