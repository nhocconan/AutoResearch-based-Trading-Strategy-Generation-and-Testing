#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 1w trend filter
# - Williams Alligator (13,8,5 SMAs with shifts) defines trend on 4h
# - Long when jaw < teeth < lips (bullish alignment), short when jaw > teeth > lips (bearish)
# - Entry confirmed by 1d volume > 1.8x 20-period average for conviction
# - Exit on opposite Alligator alignment or volume drop
# - Session filter: 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_WilliamsAlligator_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator
    df_4h = get_htf_data(prices, '4h')
    
    # Williams Alligator on 4h: SMAs with specific shifts
    # Jaw: 13-period SMA shifted 8 bars
    jaw_raw = pd.Series(df_4h['close'].values).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    # Teeth: 8-period SMA shifted 5 bars
    teeth_raw = pd.Series(df_4h['close'].values).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    # Lips: 5-period SMA shifted 3 bars
    lips_raw = pd.Series(df_4h['close'].values).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Align Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips.values)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter (only use if available)
    try:
        df_1w = get_htf_data(prices, '1w')
        close_1w = df_1w['close'].values
        # Simple trend: price > 50-period SMA on weekly
        ma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
        ma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ma_50_1w)
        use_weekly_filter = True
    except:
        # If weekly data not available, disable filter
        ma_50_1w_aligned = np.ones(n)  # neutral values
        use_weekly_filter = False
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or (use_weekly_filter and np.isnan(ma_50_1w_aligned[i]))):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.8x 1d average volume (scaled to 4h)
        # Approximate: 1d volume / 6 (since 6x4h in 1d) gives expected 4h volume
        expected_4h_vol = vol_ma_1d_aligned[i] / 6.0
        volume_filter = expected_4h_vol > 0 and volume[i] > 1.8 * expected_4h_vol
        
        # Weekly trend filter: only trade in direction of weekly trend
        if use_weekly_filter:
            weekly_uptrend = close[i] > ma_50_1w_aligned[i]
            weekly_downtrend = close[i] < ma_50_1w_aligned[i]
        else:
            weekly_uptrend = True  # neutral when filter disabled
            weekly_downtrend = True
        
        if position == 0:
            # Look for long entry: bullish Alligator alignment + volume + weekly uptrend
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                volume_filter and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish Alligator alignment + volume + weekly downtrend
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                  volume_filter and weekly_downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish Alligator alignment or volume drop
            if (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] or
                not volume_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish Alligator alignment or volume drop
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] or
                not volume_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals