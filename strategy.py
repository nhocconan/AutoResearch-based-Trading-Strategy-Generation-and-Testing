#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend absence/presence
# Only trade when all three lines are aligned (trending market) in direction of 1d EMA34
# Volume spike (1.8x 30-period average) confirms institutional participation
# Works in bull markets via buying alignments above EMA34 and bear markets via selling alignments below EMA34
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Williams_Alligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Smoothed Medians (5, 8, 13 periods) with offsets (3, 5, 8)
    # Jaw (blue line): 13-period Smoothed Median, offset 8 bars
    jaw_period = 13
    jaw_offset = 8
    median_jaw = pd.Series(high).rolling(window=jaw_period, min_periods=jaw_period).apply(
        lambda x: np.median(x), raw=True
    ).values
    jaw = pd.Series(median_jaw).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, jaw_offset)
    jaw[:jaw_offset] = np.nan
    
    # Teeth (red line): 8-period Smoothed Median, offset 5 bars
    teeth_period = 8
    teeth_offset = 5
    median_teeth = pd.Series(high).rolling(window=teeth_period, min_periods=teeth_period).apply(
        lambda x: np.median(x), raw=True
    ).values
    teeth = pd.Series(median_teeth).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, teeth_offset)
    teeth[:teeth_offset] = np.nan
    
    # Lips (green line): 5-period Smoothed Median, offset 3 bars
    lips_period = 5
    lips_offset = 3
    median_lips = pd.Series(high).rolling(window=lips_period, min_periods=lips_period).apply(
        lambda x: np.median(x), raw=True
    ).values
    lips = pd.Series(median_lips).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, lips_offset)
    lips[:lips_offset] = np.nan
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, jaw_offset + jaw_period, teeth_offset + teeth_period, lips_offset + lips_period, 30)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Lips > Teeth > Jaw (aligned up) AND above 1d EMA34 (uptrend)
                if curr_lips > curr_teeth and curr_teeth > curr_jaw and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Lips < Teeth < Jaw (aligned down) AND below 1d EMA34 (downtrend)
                elif curr_lips < curr_teeth and curr_teeth < curr_jaw and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Alligator lines cross (trend ending) or close below Teeth
            if curr_lips < curr_teeth or curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator lines cross (trend ending) or close above Teeth
            if curr_lips > curr_teeth or curr_close > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals