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
    
    # Get daily data for Williams Alligator and Elder Ray components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5 SMAs)
    close_1d_series = pd.Series(df_1d['close'].values)
    jaw = close_1d_series.rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = close_1d_series.rolling(window=8, min_periods=8).mean().values   # Red line (8)
    lips = close_1d_series.rolling(window=5, min_periods=5).mean().values   # Green line (5)
    
    # Elder Ray Components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Williams Alligator: Mouth open (teeth > lips for bullish, teeth < lips for bearish)
        bullish_alligator = teeth_aligned[i] > lips_aligned[i]
        bearish_alligator = teeth_aligned[i] < lips_aligned[i]
        
        # Elder Ray: Bull power > 0 and Bear power < 0 for trend strength
        strong_bull = bull_power_aligned[i] > 0
        strong_bear = bear_power_aligned[i] < 0
        
        # Entry conditions: 
        # Long: Alligator bullish + Elder Ray bullish + volume
        # Short: Alligator bearish + Elder Ray bearish + volume
        long_entry = bullish_alligator and strong_bull and vol_filter
        short_entry = bearish_alligator and strong_bear and vol_filter
        
        # Exit conditions: Alligator mouth closes (teeth crosses lips) or Elder Ray weakens
        long_exit = (not bullish_alligator) or (bull_power_aligned[i] <= 0)
        short_exit = (not bearish_alligator) or (bear_power_aligned[i] >= 0)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_Alligator_Volume_Session"
timeframe = "6h"
leverage = 1.0