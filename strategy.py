#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when Jaw > Teeth > Lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when alignment breaks (Jaw-Teeth-Lips order disrupted)
# Williams Alligator identifies trend phases; EMA50 filters direction; volume confirms strength
# Target: 60-120 total trades over 4 years (15-30/year) for 60-240 max total

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Williams Alligator on 6h data (Smoothed Medians)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate median (HL/2) for smoothing
    median_price = (high + low) / 2
    
    # Jaw (Blue line) - 13-period SMMA shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, jaw_shift)
    jaw[:jaw_shift] = np.nan
    
    # Teeth (Red line) - 8-period SMMA shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips (Green line) - 5-period SMMA shifted 3 bars
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, lips_shift)
    lips[:lips_shift] = np.nan
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 on 1d close
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_shift + jaw_period, teeth_shift + teeth_period, lips_shift + lips_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
            
            # Long conditions: bullish alignment, price > EMA50, volume spike
            long_cond = bullish_alignment and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: bearish alignment, price < EMA50, volume spike
            short_cond = bearish_alignment and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bullish alignment breaks
            if not ((jaw[i] < teeth[i]) and (teeth[i] < lips[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bearish alignment breaks
            if not ((jaw[i] > teeth[i]) and (teeth[i] > lips[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals