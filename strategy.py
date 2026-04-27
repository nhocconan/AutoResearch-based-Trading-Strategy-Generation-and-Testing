#!/usr/bin/env python3
"""
1d_Alligator_Trend_WeeklyEMA50_VolumeSpike
Hypothesis: 1d Williams Alligator (jaw/teeth/lips) with weekly EMA50 trend filter and volume spike confirmation.
Long when Alligator is bullish (lips > teeth > jaw) AND price > weekly EMA50 AND volume spike (>2x avg).
Short when Alligator is bearish (lips < teeth < jaw) AND price < weekly EMA50 AND volume spike.
Exit when Alligator loses alignment or weekly EMA50 alignment breaks.
Designed for 20-50 trades/year on 1d to minimize fee drag while capturing strong daily trends aligned with weekly trend.
Works in bull markets (Alligator bullish with weekly uptrend) and bear markets (Alligator bearish with weekly downtrend).
Alligator uses SMAs: jaw=13, teeth=8, lips=5 with offsets 8,5,3 respectively.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator from 1d data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Alligator components: SMAs with specific periods and offsets
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_offset).values
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_offset).values
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_offset).values
    
    # Align Alligator components to 1d timeframe (no additional delay needed for SMAs)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Alligator (jaw: 13+8=21), weekly EMA50 (~50 weeks), volume avg
    start_idx = max(50, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment conditions
        bullish_alignment = (lips_val > teeth_val and teeth_val > jaw_val)
        bearish_alignment = (lips_val < teeth_val and teeth_val < jaw_val)
        
        if position == 0:
            # Flat - look for entry: Alligator alignment with weekly EMA50 and volume spike
            # Long: Bullish alignment AND price > weekly EMA50 AND volume spike
            # Short: Bearish alignment AND price < weekly EMA50 AND volume spike
            long_condition = bullish_alignment and (close_val > ema_val) and vol_spike
            short_condition = bearish_alignment and (close_val < ema_val) and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when Alligator loses bullish alignment OR loses weekly EMA50 alignment
            if not bullish_alignment or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Alligator loses bearish alignment OR loses weekly EMA50 alignment
            if not bearish_alignment or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Alligator_Trend_WeeklyEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0