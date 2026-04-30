#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h EMA50 trend filter + volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trendless periods and trade only when aligned
# Only trade when Alligator lines are bent (trending) and price is outside mouth in EMA50 trend direction
# Volume spike (1.8x 24-period average) confirms participation
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Williams_Alligator_12hEMA50_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator (13,8,5) - Smoothed Median Price
    # Median Price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw_period = 13
    jaw_shift = 8
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, jaw_shift)
    jaw[:jaw_shift] = np.nan
    
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth_period = 8
    teeth_shift = 5
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips (5-period SMMA, shifted 3 bars)
    lips_period = 5
    lips_shift = 3
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, lips_shift)
    lips[:lips_shift] = np.nan
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, jaw_shift + jaw_period, teeth_shift + teeth_period, lips_shift + lips_period, 24)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        
        # Alligator trend condition: lines must be bent (not intertwined)
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_aligned = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_aligned = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Alligator aligned
            if curr_volume_spike:
                # Bullish entry: price above Lips AND above 12h EMA50 (uptrend) AND bullish alignment
                if curr_close > curr_lips and curr_close > curr_ema_50_12h and bullish_aligned:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price below Lips AND below 12h EMA50 (downtrend) AND bearish alignment
                elif curr_close < curr_lips and curr_close < curr_ema_50_12h and bearish_aligned:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Lips or Alligator loses bullish alignment
            if curr_close < curr_lips or not (curr_lips > curr_teeth and curr_teeth > curr_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Lips or Alligator loses bearish alignment
            if curr_close > curr_lips or not (curr_lips < curr_teeth and curr_teeth < curr_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals