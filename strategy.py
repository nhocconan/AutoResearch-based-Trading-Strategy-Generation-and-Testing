#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w trend filter + volume confirmation
# Uses Alligator (Jaw/Teeth/Lips) to detect trends, only trades in direction of 1w EMA50
# Volume spike confirms breakout strength. Designed for low-frequency, high-conviction trades.
# Works in bull (rides trends) and bear (avoids false signals via 1w filter).
name = "12h_WilliamsAlligator_1wTrend_Volume"
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
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams Alligator (SMAs on median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components from 1d median price (HL/2)
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # Blue line (13-bar SMA, 8 bars ahead)
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values    # Red line (8-bar SMA, 5 bars ahead)
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values     # Green line (5-bar SMA, 3 bars ahead)
    
    # Shift jaws/teeth/lips to avoid look-ahead (Williams Alligator uses future-shifted SMAs)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set initial values to NaN where roll creates invalid data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for 1w EMA and Alligator
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_50_1w_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
        is_uptrend = lips_val > teeth_val > jaw_val
        is_downtrend = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Enter long: Alligator uptrend + price above lips + 1w uptrend + volume spike
            if is_uptrend and close[i] > lips_val and close[i] > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator downtrend + price below lips + 1w downtrend + volume spike
            elif is_downtrend and close[i] < lips_val and close[i] < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns down (lips < jaw) or price below teeth
            if lips_val < jaw_val or close[i] < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns up (lips > jaw) or price above teeth
            if lips_val > jaw_val or close[i] > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals