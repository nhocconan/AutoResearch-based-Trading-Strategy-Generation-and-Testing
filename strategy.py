#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsAlligator_ElderRay_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Williams Alligator (13,8,5 SMAs on median price)
    df_1w = get_htf_data(prices, '1w')
    median_1w = (df_1w['high'] + df_1w['low']) / 2
    jaw = median_1w.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = median_1w.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = median_1w.rolling(window=5, min_periods=5).mean().shift(3).values
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Load daily data for Elder Ray (13-period EMA)
    df_1d = get_htf_data(prices, '1d')
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (df_1d['high'] - ema13_1d).values
    bear_power = (df_1d['low'] - ema13_1d).values
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: current volume > 1.8 * 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma_30)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_filter = volume_filter[i]
        sess_filter = session_filter[i]
        
        # Alligator aligned: jaws < teeth < lips (bullish) or jaws > teeth > lips (bearish)
        bullish_alligator = (jaw_val < teeth_val) and (teeth_val < lips_val)
        bearish_alligator = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        if position == 0:
            # Long: bullish Alligator + positive Bull Power + volume + session
            if bullish_alligator and (bull_val > 0) and vol_filter and sess_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + negative Bear Power + volume + session
            elif bearish_alligator and (bear_val < 0) and vol_filter and sess_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power turns negative
            if not bullish_alligator or (bull_val <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power turns positive
            if not bearish_alligator or (bear_val >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals