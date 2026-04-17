#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian channel breakout (20-period) + volume confirmation + 1d ADX trend filter.
Long when price breaks above 1d Donchian upper channel with volume confirmation and 1d ADX > 25 (strong uptrend).
Short when price breaks below 1d Donchian lower channel with volume confirmation and 1d ADX > 25 (strong downtrend).
Exit when price returns to the 1d Donchian midpoint or reverses with volume.
Uses 1d timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture strong trends with institutional volume while avoiding false breakouts in ranging markets.
Donchian channels provide clear trend-following signals based on recent price extremes.
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
    
    # Get 1d data for Donchian and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    period = 20
    high_roll = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    upper_chan = high_roll
    lower_chan = low_roll
    midpoint_chan = (upper_chan + lower_chan) / 2.0
    
    # Calculate 1d ADX (14-period) for trend filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_series.diff()
    down_move = low_series.shift(1) - low_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_chan_aligned = align_htf_to_ltf(prices, df_1d, upper_chan)
    lower_chan_aligned = align_htf_to_ltf(prices, df_1d, lower_chan)
    midpoint_chan_aligned = align_htf_to_ltf(prices, df_1d, midpoint_chan)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_chan_aligned[i]) or 
            np.isnan(lower_chan_aligned[i]) or 
            np.isnan(midpoint_chan_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper channel with volume and strong uptrend (ADX > 25)
            if (close[i] > upper_chan_aligned[i] and 
                volume_confirmed and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower channel with volume and strong downtrend (ADX > 25)
            elif (close[i] < lower_chan_aligned[i] and 
                  volume_confirmed and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below lower channel with volume (reversal)
            if (close[i] <= midpoint_chan_aligned[i] or 
                (close[i] < lower_chan_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above upper channel with volume (reversal)
            if (close[i] >= midpoint_chan_aligned[i] or 
                (close[i] > upper_chan_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Breakout_Volume_ADX25_Trend"
timeframe = "4h"
leverage = 1.0