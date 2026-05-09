#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when price above Alligator jaws/teeth/lips (aligned) with 1d EMA50 uptrend and volume spike
# Short when price below Alligator jaws/teeth/lips (aligned) with 1d EMA50 downtrend and volume spike
# Exit when price crosses back through Alligator teeth or reverses to opposite lip
# Uses Alligator for trend identification, EMA for higher timeframe trend, volume for conviction
# Designed to capture trends with controlled frequency in both bull and bear markets
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1h data for Alligator (SMAs of median price)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 13:
        return np.zeros(n)
    
    # Calculate median price
    median_price = (df_1h['high'] + df_1h['low']) / 2
    
    # Williams Alligator lines (13, 8, 5 period SMAs with future shifts)
    jaws = median_price.rolling(window=13, min_periods=13).mean().shift(8)   # Blue line
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5)    # Red line
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3)     # Green line
    
    # Align Alligator lines to 4h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1h, jaws.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1h, lips.values)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above all Alligator lines, EMA50 uptrend, volume spike
            if (close[i] > jaws_aligned[i] and 
                close[i] > teeth_aligned[i] and 
                close[i] > lips_aligned[i] and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below all Alligator lines, EMA50 downtrend, volume spike
            elif (close[i] < jaws_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  close[i] < lips_aligned[i] and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below teeth or reverses to lips
            if (close[i] < teeth_aligned[i]) or (close[i] < lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above teeth or reverses to lips
            if (close[i] > teeth_aligned[i]) or (close[i] > lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals