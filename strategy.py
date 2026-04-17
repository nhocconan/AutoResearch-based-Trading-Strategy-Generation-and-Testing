#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Williams Alligator trend filter + 1d Camarilla pivot breakout + volume confirmation.
Long when price breaks above Camarilla R3 with bullish Alligator alignment (jaw < teeth < lips) and volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 with bearish Alligator alignment (jaw > teeth > lips) and volume > 1.5x 20-period average.
Williams Alligator identifies trend structure with minimal lag, Camarilla provides precise intraday support/resistance levels, volume confirms breakout validity.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation) by requiring trend alignment.
Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (trend)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 12h
    jaw, teeth, lips = compute_williams_alligator(
        high_12h, low_12h, close_12h
    )
    
    # Calculate Camarilla levels on 1d
    def camarilla_levels(high, low, close):
        """Calculate Camarilla pivot levels: R3, R2, R1, PP, S1, S2, S3"""
        pp = (high + low + close) / 3
        range_ = high - low
        r1 = pp + range_ * 1.1 / 12
        r2 = pp + range_ * 1.1 / 6
        r3 = pp + range_ * 1.1 / 4
        s1 = pp - range_ * 1.1 / 12
        s2 = pp - range_ * 1.1 / 6
        s3 = pp - range_ * 1.1 / 4
        return r3, r2, r1, pp, s1, s2, s3
    
    camarilla_r3, camarilla_r2, camarilla_r1, camarilla_pp, camarilla_s1, camarilla_s2, camarilla_s3 = camarilla_levels(
        high_1d, low_1d, close_1d
    )
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # Alligator trend: bullish (jaw < teeth < lips) or bearish (jaw > teeth > lips)
        bullish_alligator = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alligator = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with bullish Alligator and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                bullish_alligator and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with bearish Alligator and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  bearish_alligator and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla S3 (opposite side)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla R3 (opposite side)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hAlligator_1dCamarilla_R3S3_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0