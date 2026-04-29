#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND volume > 2.0x 20-bar avg
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND volume > 2.0x 20-bar avg
# Exit when Alligator alignment breaks OR Elder power reverses
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Combines trend-following (Alligator) with momentum (Elder Ray) and volume confirmation
# to capture strong moves while avoiding false signals in choppy markets.
# Novelty: Uses 12h primary timeframe (less crowded than 4h/6h) with proven Williams Alligator and Elder Ray indicators
# that work well in both bull and bear markets by adapting to trend strength and momentum.

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5 SMAs with 8,5,3 shifts)
    # Jaws: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars
    # Lips: 5-period SMA shifted 3 bars
    jaws_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Index (13-period EMA of high and low)
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    ema_13_close = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_close
    bear_power_1d = low_1d - ema_13_close
    
    # Align all HTF indicators to 12h timeframe
    jaws_12h = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_12h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_12h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8)  # volume MA and Alligator warmup (13+8=21 for jaws)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or 
            np.isnan(bull_power_12h[i]) or np.isnan(bear_power_12h[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_jaws = jaws_12h[i]
        curr_teeth = teeth_12h[i]
        curr_lips = lips_12h[i]
        curr_bull = bull_power_12h[i]
        curr_bear = bear_power_12h[i]
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_alignment = curr_jaws < curr_teeth < curr_lips
        bearish_alignment = curr_jaws > curr_teeth > curr_lips
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator bullish alignment breaks OR Elder Bull Power turns negative
            if not bullish_alignment or curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator bearish alignment breaks OR Elder Bear Power turns positive
            if not bearish_alignment or curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Alligator bullish alignment AND Elder Bull Power > 0 AND volume confirmation
            if bullish_alignment and curr_bull > 0 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator bearish alignment AND Elder Bear Power < 0 AND volume confirmation
            elif bearish_alignment and curr_bear < 0 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals