#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate 6-week EMA for trend filter
    ema_6_1w = pd.Series(close_1w).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema_6_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_6_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 6h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema_6_1w_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(atr_6[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA6
        price_above_ema = close[i] > ema_6_1w_aligned[i]
        price_below_ema = close[i] < ema_6_1w_aligned[i]
        
        # Volume filter: current volume above daily average
        volume_filter = volume[i] > volume_ma_1d_aligned[i] * 1.2
        
        # Volatility filter: avoid extremely high volatility
        atr_median = np.nanmedian(atr_6[max(0, i-50):i+1]) if i >= 10 else atr_6[i]
        volatility_filter = atr_6[i] < atr_median * 2.0
        
        # Price relative to weekly pivot levels
        at_pivot = abs(close[i] - pivot_1w_aligned[i]) < (r1_1w_aligned[i] - s1_1w_aligned[i]) * 0.05
        near_r1 = close[i] > r1_1w_aligned[i] * 0.98 and close[i] < r1_1w_aligned[i] * 1.02
        near_s1 = close[i] > s1_1w_aligned[i] * 0.98 and close[i] < s1_1w_aligned[i] * 1.02
        near_r2 = close[i] > r2_1w_aligned[i] * 0.98 and close[i] < r2_1w_aligned[i] * 1.02
        near_s2 = close[i] > s2_1w_aligned[i] * 0.98 and close[i] < s2_1w_aligned[i] * 1.02
        near_r3 = close[i] > r3_1w_aligned[i] * 0.98 and close[i] < r3_1w_aligned[i] * 1.02
        near_s3 = close[i] > s3_1w_aligned[i] * 0.98 and close[i] < s3_1w_aligned[i] * 1.02
        breakout_r4 = close[i] > r4_1w_aligned[i]
        breakdown_s4 = close[i] < s4_1w_aligned[i]
        
        # Long conditions: 
        # 1. Bounce from support (S1, S2, S3) with volume in uptrend
        # 2. Breakout above R4 with volume in uptrend
        long_bounce = ((near_s1 or near_s2 or near_s3) and price_above_ema and volume_filter and volatility_filter)
        long_breakout = (breakout_r4 and price_above_ema and volume_filter and volatility_filter)
        long_condition = long_bounce or long_breakout
        
        # Short conditions: 
        # 1. Rejection from resistance (R1, R2, R3) with volume in downtrend
        # 2. Breakdown below S4 with volume in downtrend
        short_rejection = ((near_r1 or near_r2 or near_r3) and price_below_ema and volume_filter and volatility_filter)
        short_breakdown = (breakdown_s4 and price_below_ema and volume_filter and volatility_filter)
        short_condition = short_rejection or short_breakdown
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: 
        # - Reverse signal
        # - Price returns to pivot area
        elif position == 1 and (short_condition or at_pivot):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (long_condition or at_pivot):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_BounceBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0