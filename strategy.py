#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Volume Spike + 12h EMA34 Trend Filter
Long when Alligator jaws (13) < teeth (8) < lips (5) AND 1d volume > 2x 20-day average AND 12h EMA34 rising.
Short when Alligator jaws > teeth > lips AND 1d volume > 2x 20-day average AND 12h EMA34 falling.
Exit when Alligator lines re-cross (jaws > teeth or teeth > lips) OR 12h EMA34 flips direction.
Uses Alligator for trend identification, volume spike for confirmation, and 12h EMA for higher timeframe alignment.
Designed to catch strong trends with institutional participation while filtering chop. Target: 12-25 trades/year per symbol.
"""

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
    
    # Get 6h data for Williams Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator on 6h close
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    jaw_6h = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_6h = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_6h = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d volume spike (> 2x 20-day average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Calculate 12h EMA34 and its direction
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_prev = np.roll(ema_34_12h, 1)
    ema_34_12h_prev[0] = np.nan
    ema_rising = ema_34_12h > ema_34_12h_prev
    ema_falling = ema_34_12h < ema_34_12h_prev
    
    # Align all indicators to primary 6h timeframe
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_6h_aligned[i]) or 
            np.isnan(teeth_6h_aligned[i]) or
            np.isnan(lips_6h_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        jaw = jaw_6h_aligned[i]
        teeth = teeth_6h_aligned[i]
        lips = lips_6h_aligned[i]
        
        alligator_long = (jaw < teeth) and (teeth < lips)  # jaws < teeth < lips (bullish alignment)
        alligator_short = (jaw > teeth) and (teeth > lips)  # jaws > teeth > lips (bearish alignment)
        alligator_exit = not alligator_long and not alligator_short  # lines crossed or tangled
        
        # Volume confirmation and trend filter
        volume_confirmed = volume_spike_aligned[i] == 1.0
        trend_up = ema_rising_aligned[i] == 1.0
        trend_down = ema_falling_aligned[i] == 1.0
        
        if position == 0:
            # Long: bullish Alligator alignment + volume spike + rising 12h EMA34
            if alligator_long and volume_confirmed and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + volume spike + falling 12h EMA34
            elif alligator_short and volume_confirmed and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR 12h EMA34 falls
            if (alligator_exit or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR 12h EMA34 rises
            if (alligator_exit or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dVolumeSpike_12hEMA34_Trend"
timeframe = "6h"
leverage = 1.0