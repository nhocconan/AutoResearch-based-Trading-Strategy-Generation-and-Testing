#!/usr/bin/env python3
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
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Williams Alligator (13,8,5) - trend identification
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Red line (8)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values   # Green line (5)
    
    # 1d ADX(14) - trend strength
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    
    # 1d Volume ratio - volume confirmation
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_1d / vol_ma
    vol_ratio_values = vol_ratio.values
    
    # Align HTF indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: jaws < teeth < lips = bearish, jaws > teeth > lips = bullish
        bullish_alligator = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        bearish_alligator = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio_aligned[i] > 1.5
        
        # Entry conditions
        long_entry = bullish_alligator and strong_trend and volume_confirm
        short_entry = bearish_alligator and strong_trend and volume_confirm
        
        # Exit conditions: loss of alignment or weak trend
        long_exit = not bullish_alligator or adx_aligned[i] < 20
        short_exit = not bearish_alligator or adx_aligned[i] < 20
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_ADX_Volume_Confirm"
timeframe = "6h"
leverage = 1.0