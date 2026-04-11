#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 34 or len(df_1d) < 34:
        return signals
    
    # Williams Alligator on 12h: 3 SMAs (Jaw=13, Teeth=8, Lips=5) with future shift
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2.0
    
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    jaw_12h = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)
    jaw_12h[:8] = np.nan
    
    # Teeth (red): 8-period SMMA, shifted 5 bars forward
    teeth_12h = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    teeth_12h[:5] = np.nan
    
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    lips_12h = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)
    lips_12h[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 1d ADX for trend strength filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        # Alligator sleeping: jaws, teeth, lips intertwined (no clear trend)
        # Alligator awakening: lines diverge in specific order
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw (green < red < blue)
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Entry signals
        long_signal = bullish_align and strong_trend and volume_confirmed
        short_signal = bearish_align and strong_trend and volume_confirmed
        
        # Exit when Alligator starts to sleep again (lines re-intertwine)
        # Or when trend weakens (ADX < 20)
        exit_condition = (
            (abs(lips_val - teeth_val) < 0.001 * close[i]) or  # Lines close together
            (adx_aligned[i] < 20)
        )
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position != 0 and exit_condition:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Williams Alligator on 12h with ADX filter on 1d for 6h timeframe.
# The Alligator identifies trend formation and direction: 
# - Bullish when Lips > Teeth > Jaw (green > red > blue)
# - Bearish when Lips < Teeth < Jaw (green < red < blue)
# ADX > 25 on 1d ensures we only trade in strong trending markets.
# Volume confirmation (>1.3x average) filters out low-credibility breakouts.
# Exits when the Alligator goes back to sleep (lines intertwine) or trend weakens (ADX < 20).
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag.
# Works in both bull and bear markets by trading the direction of the trend once established.