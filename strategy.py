#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d ADX trend filter and volume confirmation
# Long when Alligator jaws (13-period SMMA) above teeth (8-period SMMA) and lips (5-period SMMA) + ADX > 25 + volume spike
# Short when jaws below teeth and lips + ADX > 25 + volume spike
# Exit when Alligator lines converge (jaws between teeth and lips) or ADX < 20
# Designed for low trade frequency (~10-30/year) with strong trend-following edge in both bull and bear markets
# Uses Williams Alligator for trend identification and ADX for trend strength confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (14-period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = WilderSmooth(tr, 14)
    plus_dm14 = WilderSmooth(plus_dm, 14)
    minus_dm14 = WilderSmooth(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 12h timeframe
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Smoothed Moving Average (SMMA) - Wilder's smoothing
    def SMMA(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13, 8), Teeth (8, 5), Lips (5, 3)
    jaw = SMMA(SMMA(high, 13) + SMMA(low, 13), 2) / 2  # (SMMA(H,13) + SMMA(L,13))/2
    teeth = SMMA(SMMA(high, 8) + SMMA(low, 8), 2) / 2
    lips = SMMA(SMMA(high, 5) + SMMA(low, 5), 2) / 2
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:19] = np.nan  # Not enough data for first 19 periods
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        # Alligator conditions
        jaws_above_teeth = jaw_val > teeth_val
        teeth_above_lips = teeth_val > lips_val
        jaws_below_teeth = jaw_val < teeth_val
        teeth_below_lips = teeth_val < lips_val
        
        # Convergence (jaws between teeth and lips)
        converged = (jaw_val > teeth_val and jaw_val < lips_val) or \
                   (jaw_val < teeth_val and jaw_val > lips_val)
        
        if position == 0:
            # Long conditions: jaws > teeth > lips + ADX > 25 + volume spike
            if jaws_above_teeth and teeth_above_lips and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: jaws < teeth < lips + ADX > 25 + volume spike
            elif jaws_below_teeth and teeth_below_lips and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator lines converge or ADX < 20
            exit_signal = False
            
            if converged or adx_val < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dADX_Volume"
timeframe = "12h"
leverage = 1.0